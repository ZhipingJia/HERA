import os
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from typing import List
import random
import time
import lm_eval
from lm_eval.models.huggingface import LoadedHFLM
from lm_eval.utils import (
    run_task_tests,
    get_git_commit_hash,
)
from lm_eval.evaluator import evaluate
from lm_eval.logger import eval_logger

IGNORE_INDEX = -100


class BaseEvalCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        tokenizer,
        eval_steps=None,
        eval_start=None,
        do_init_eval=False,
        do_final_eval=False,
    ) -> None:
        """base evaluation callback to control when to do the evaluation.

        Args:
            trainer (Trainer): _description_
            tokenizer (_type_): _description_
            eval_steps (_type_, optional): eval interval. Defaults to None.
            eval_start (_type_, optional): which step to start eval. Defaults to None.
            do_init_eval (bool, optional): eval before model training. Defaults to False.
        """
        if eval_steps is None:
            eval_steps = 1
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.eval_start = eval_start if eval_start is not None else 0
        self.do_init_eval = do_init_eval
        self.do_final_eval = do_final_eval
        self._last_eval_step = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0 and self.do_init_eval:
            self.evaluate(args, state, control, **kwargs)
            self._last_eval_step = state.global_step
        if (
            state.global_step % self.eval_steps == 0
            and state.global_step != 0
            and state.global_step >= self.eval_start
        ):
            self.evaluate(args, state, control, **kwargs)
            self._last_eval_step = state.global_step

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (
            self.do_final_eval
            and state.global_step > 0
            and self._last_eval_step != state.global_step
        ):
            self.evaluate(args, state, control, **kwargs)
            self._last_eval_step = state.global_step

    def evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass


class EvalHarnessCallBack(BaseEvalCallback):
    def __init__(
        self,
        trainer: Trainer,
        tokenizer,
        tasks: List[str],
        eval_steps=None,
        eval_start=None,
        do_init_eval=False,
        do_final_eval=False,
        eval_batch_size=32,
        eval_limit=None,
        log_samples=True,
    ) -> None:
        """This callback integrates Eleuther/lm-evaluation-harness into the training loop

        Args:
            trainer (Trainer): trainer
            tokenizer (_type_): tokenizer
            tasks (List[str]): evaluation task name, pls refer to yaml files of lm-evaluation-harness.
            eval_steps (_type_, optional): eval interval. Defaults to None.
            eval_start (_type_, optional): which step to start eval. Defaults to None.
            do_init_eval (bool, optional): eval before model training. Defaults to False.
            eval_batch_size (int, optional):  Defaults to 32.

        """
        super().__init__(
            trainer,
            tokenizer,
            eval_steps,
            eval_start,
            do_init_eval,
            do_final_eval,
        )
        self.tasks = tasks
        self.eval_batch_size = eval_batch_size
        self.log_samples = log_samples  # 初始化 log_samples 参数
        self.eval_limit=eval_limit

    def evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        start_time = time.time()  # 记录评估开始的时间

        # 1. 模型进入评估模式
        model_eval_start = time.time()
        self.trainer.model.eval()
        model_eval_end = time.time()
        print(f"Switching to eval mode took: {model_eval_end - model_eval_start:.2f} seconds")


        # 2. 加载 HFLM 模型
        hflm_load_start = time.time()
        lm = LoadedHFLM(
            model=self.trainer.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            max_batch_size=128,
        )
        hflm_load_end = time.time()
        print(f"Loading HFLM model took: {hflm_load_end - hflm_load_start:.2f} seconds")

        # 3. 执行评估
        eval_start = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=self.tasks,
            # num_fewshot=0,
            use_cache=None,
            limit=self.eval_limit,
            log_samples=self.log_samples,
            # limit=self.eval_batch_size*2,
        )
        # print('res:')
        # print(res)
        eval_end = time.time()
        print(f"Running evaluation took: {eval_end - eval_start:.2f} seconds")
        if args.local_rank == 0:
            log_start = time.time()
            self.trainer.log(self.format_metrics_for_tb(res["results"]))
            log_end = time.time()
            print(f"Logging results took: {log_end - log_start:.2f} seconds")
            print("trainer log done")
        
        # 恢复模型到训练模式
        model_train_start = time.time()
        self.trainer.model.train()
        model_train_end = time.time()
        print(f"Switching back to train mode took: {model_train_end - model_train_start:.2f} seconds")
        # 5. 评估过程结束
        end_time = time.time()
        print(f"Total evaluation process took: {end_time - start_time:.2f} seconds")
        print("evaluate done")

    def format_metrics_for_tb(self, results):
        res = {}
        for task, metrics in results.items():
            for metric_name, value in metrics.items():
                res[f"{task}-{metric_name}"] = value
        return res


def filter_private_lora_state_dict(
    state_dict,
    save_lm_head_lora: bool = False,
    save_embed_tokens: bool = False,
):
    to_save = {}
    for k, v in state_dict.items():
        should_save = False

        if "lora_" in k:
            if save_lm_head_lora or "lora_lm_head" not in k:
                should_save = True

        if save_embed_tokens and "embed_tokens" in k:
            should_save = True

        if should_save:
            to_save[k] = v.cpu()
    return to_save


class PrivateLoraPeriodicSaveCallBack(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        save_steps: int,
        save_lm_head_lora: bool = False,
        save_embed_tokens: bool = False,
    ):
        self.trainer = trainer
        self.save_steps = int(save_steps)
        self.save_lm_head_lora = save_lm_head_lora
        self.save_embed_tokens = save_embed_tokens
        self._saved_steps = set()

    def _collect_state_dict(self, step):
        state_dict = None
        if hasattr(self.trainer.args, "hf_deepspeed_config") and self.trainer.args.hf_deepspeed_config.is_zero3():
            if self.trainer.is_world_process_zero():
                print(f"[PeriodicSave] Consolidating Zero3 state dict for step {step}...")
            state_dict = self.trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        else:
            if self.trainer.is_world_process_zero():
                state_dict = self.trainer.model.state_dict()
        return state_dict

    def _save(self, args: TrainingArguments, step: int, suffix: str = None):
        if self.save_steps <= 0 or step <= 0:
            return
        if step in self._saved_steps and suffix is None:
            return

        state_dict = self._collect_state_dict(step)
        if self.trainer.is_world_process_zero() and state_dict is not None:
            folder_name = f"checkpoint-{step}" if suffix is None else f"checkpoint-{step}-{suffix}"
            save_path = os.path.join(args.output_dir, folder_name)
            os.makedirs(save_path, exist_ok=True)
            print(
                "[PeriodicSave] Filtering weights "
                f"save_lm_head_lora={self.save_lm_head_lora}, "
                f"save_embed_tokens={self.save_embed_tokens}..."
            )
            to_save = filter_private_lora_state_dict(
                state_dict,
                save_lm_head_lora=self.save_lm_head_lora,
                save_embed_tokens=self.save_embed_tokens,
            )
            if to_save:
                torch.save(to_save, f"{save_path}/pl.bin")
                print(f"[PeriodicSave] Successfully saved to: {save_path}/pl.bin")
            else:
                print("[PeriodicSave] Warning: No matching parameters found to save!")

        self._saved_steps.add(step)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_steps > 0 and state.global_step > 0 and state.global_step % self.save_steps == 0:
            self._save(args, state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_steps > 0 and state.global_step > 0:
            self._save(args, state.global_step, suffix="final")


# =============================================================================
# 请将此代码添加到 infras/eval.py 文件末尾
# =============================================================================

class EvalHarnessAndSaveCallBack(EvalHarnessCallBack):
    def __init__(
        self, 
        *args, 
        save_embedding_flag: bool = False, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_embedding_flag = save_embedding_flag

    # 修改点：增加了 output_dir 参数，显式传入路径
    def save_pl_checkpoint(self, results, step, output_dir):
        """
        保存逻辑
        """
        # --- 1. 获取准确率用于文件夹命名 ---
        acc_val = 0.0
        try:
            for task, metrics in results.items():
                for m_key, m_val in metrics.items():
                    if "acc" in m_key:
                        acc_val = m_val
                        break
                if acc_val != 0.0: break
            
            if acc_val == 0.0 and len(results) > 0:
                first_task = list(results.keys())[0]
                first_metrics = results[first_task]
                if len(first_metrics) > 0:
                    acc_val = list(first_metrics.values())[0]
        except Exception as e:
            print(f"[Save] Warning: Could not extract metric for naming: {e}")

        # --- 2. 使用传入的 output_dir 构造路径 ---
        # 文件夹名例如: checkpoint-1000-acc-0.7523
        folder_name = f"checkpoint-{step}-acc-{acc_val:.4f}"
        save_path = os.path.join(output_dir, folder_name)

        # --- 3. DeepSpeed Zero3 兼容性处理 ---
        state_dict = None
        if hasattr(self.trainer.args, "hf_deepspeed_config") and self.trainer.args.hf_deepspeed_config.is_zero3():
            if self.trainer.is_world_process_zero():
                print(f"[Save] Consolidating Zero3 state dict for step {step}...")
            state_dict = self.trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        else:
            if self.trainer.is_world_process_zero():
                state_dict = self.trainer.model.state_dict()

        # --- 4. 筛选与保存 (仅 Rank 0) ---
        if self.trainer.is_world_process_zero():
            if state_dict is not None:
                os.makedirs(save_path, exist_ok=True)
                to_save = {}
                
                print(f"[Save] Filtering weights with embedding_flag={self.save_embedding_flag}...")

                to_save = filter_private_lora_state_dict(
                    state_dict,
                    save_lm_head_lora=self.save_embedding_flag,
                    save_embed_tokens=self.save_embedding_flag,
                )

                # 执行保存
                if to_save:
                    torch.save(to_save, f"{save_path}/pl.bin")
                    print(f"[Save] Successfully saved to: {save_path}/pl.bin")
                else:
                    print(f"[Save] Warning: No matching parameters found to save!")
            
        # --- 5. 进程同步 ---
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def evaluate(self, args, state, control, **kwargs):
        # args 就是 TrainingArguments，包含了 output_dir
        
        # === (原有逻辑开始) ===
        start_time = time.time()
        
        model_eval_start = time.time()
        self.trainer.model.eval()
        model_eval_end = time.time()
        print(f"Switching to eval mode took: {model_eval_end - model_eval_start:.2f} seconds")

        hflm_load_start = time.time()
        lm = LoadedHFLM(
            model=self.trainer.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            max_batch_size=128,
        )
        hflm_load_end = time.time()
        print(f"Loading HFLM model took: {hflm_load_end - hflm_load_start:.2f} seconds")

        eval_start = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=self.tasks,
            use_cache=None,
            limit=self.eval_limit,
            log_samples=self.log_samples,
        )
        eval_end = time.time()
        print(f"Running evaluation took: {eval_end - eval_start:.2f} seconds")

        if args.local_rank == 0:
            log_start = time.time()
            self.trainer.log(self.format_metrics_for_tb(res["results"]))
            log_end = time.time()
            print(f"Logging results took: {log_end - log_start:.2f} seconds")
            print("trainer log done")
        # === (原有逻辑结束) ===

        # =========================================================
        # === 插入点：调用保存函数 ===
        # 显式传入 args.output_dir，这下绝对不会错
        # =========================================================
        current_step = state.global_step

        if res is not None and "results" in res:
            results_to_save = res["results"]
        else:
            results_to_save = {} # 防止 Rank > 0 报错

        self.save_pl_checkpoint(
            results=results_to_save, 
            step=current_step, 
            output_dir=args.output_dir 
        )
        # =========================================================

        model_train_start = time.time()
        self.trainer.model.train()
        model_train_end = time.time()
        print(f"Switching back to train mode took: {model_train_end - model_train_start:.2f} seconds")
        
        end_time = time.time()
        print(f"Total evaluation process took: {end_time - start_time:.2f} seconds")
        print("evaluate done")

def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
):
    """This code snipet is taken from lm-evaluation-harness to adapt for loaded model in a training script.

    Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(
        1234
    )  # TODO: this may affect training runs that are run with evaluation mid-run.

    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    assert isinstance(model, lm_eval.api.model.LM)
    lm = model

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank" + str(lm.rank) + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            # if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue

        config = task_obj._config
        if num_fewshot is not None:
            if config["num_fewshot"] > 0:
                default_num_fewshot = config["num_fewshot"]
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )

            task_obj._config["num_fewshot"] = num_fewshot

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
    )
    if lm.rank == 0:
        # add info about the model and few shot config
        results["config"] = {
            "model": model
            if isinstance(model, str)
            else model.model.config._name_or_path,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
        }
        results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None
