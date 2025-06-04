import argparse
import csv
import math
import time
from os.path import join
import os
import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st
import numpy as np
import yaml
from torch.utils.data import Dataset
from datetime import datetime
from task_utils import get_sim_and_prior_from_sbibm
from posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
# from sampling import get_c2st, get_truncated_prior
from utils import build_train_and_test_loaders, RuntimeLimits
import torch
import logging

print(torch.cuda.is_available())  # 应返回 True
print(torch.version.cuda)  # 应显示 CUDA 版本（如 11.8）


class SbiDataset(Dataset):
    def __init__(self, theta, x):
        super(SbiDataset, self).__init__()
        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"].to(sample.device)
        std = self.standardization[label]["std"].to(sample.device)
        if not inverse:
            return (sample - mean) / std
        else:

            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


def log(message, file):
    """打印并写入日志文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 添加时间戳
    formatted_message = f"[{timestamp}] {message}"  # 格式化日志内容
    print(formatted_message)
    with open(file, 'a') as f:
        f.write(formatted_message + '\n')


def setup_logger(output_file):
    """设置日志记录器"""
    logger = logging.getLogger("run_logger")
    logger.setLevel(logging.INFO)
    # 文件处理器
    file_handler = logging.FileHandler(output_file)
    file_handler.setLevel(logging.INFO)
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def generate_dataset(benchmark, num_rounds, simulation_budget, directory_save=None):
    task = sbibm.get_task(benchmark)
    simulator, prior = get_sim_and_prior_from_sbibm(task)
    simulation_budget_round = int(simulation_budget / num_rounds)
    theta = prior(simulation_budget_round)
    x = simulator(theta)
    # if directory_save is not None:
    #     np.save(join(directory_save, 'x.npy'), x)
    #     np.save(join(directory_save, 'theta.npy'), theta)
    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)
    # x 和 theta的维度都是 ([simulation_budget, dimension])
    dataset = SbiDataset(theta, x)
    return dataset, simulation_budget_round


def load_dataset(directory_save, settings):
    x = np.load(join(directory_save, 'x.npy'))
    theta = np.load(join(directory_save, 'theta.npy'))
    x = torch.tensor(x, dtype=torch.float)
    theta = torch.tensor(theta, dtype=torch.float)
    settings["task"]["dim_theta"] = theta.shape[1]
    settings["task"]["dim_x"] = x.shape[1]
    dataset = SbiDataset(theta, x)
    return dataset


def train_model(train_dir, settings, train_loader, test_loader):
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )
    # 构建模型, 注意这里对应的是函数内的 if filename is None
    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device=settings["training"].get("device", "cuda"),
    )
    # Selected model class: <class 'posterior_models.flow_matching.FlowMatching'>

    # Before training you need to call the following lines:
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        early_stopping=True,
    )  # 大批量的输出都是从这里出现的

    # load the best model
    # 注意这里对应的是函数内的 if filename is not None
    best_model = build_model_from_kwargs(
        filename=join(train_dir, "best_model.pt"),
        device=settings["training"].get("device", "cuda"),
    )
    return best_model


def get_truncated_prior(sample_epsilon, batch_size, num_samples, observation, prior, model, logger=None):
    # est_posterior_samples, log_probs = model.sample_and_log_prob_batch(observation.repeat((1000, 1)))
    log(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB", output_file)
    log(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB", output_file)

    est_posterior_samples = []
    log_probs = []
    count = 0

    for i in range(0, num_samples, batch_size):
        obs_repeated = observation.repeat((batch_size, 1))
        samples = model.sample_batch(obs_repeated)
        est_posterior_samples.append(samples)

        # 清理临时变量并释放显存
        del obs_repeated
        torch.cuda.empty_cache()

        # 分批计算 log_prob
        log_probs.append(model.log_prob_batch(samples, observation.repeat((batch_size, 1))))
        print(
            f"Iteration {count + 1}: Generated {batch_size * (count + 1)}/{num_samples} posterior samples"
        )
        count = count + 1

    est_posterior_samples = torch.cat(est_posterior_samples)
    log_probs = torch.cat(log_probs)

    # est_posterior_samples = model.sample_batch(observation.repeat((1000, 1)))
    # log_probs = model.log_prob_batch(est_posterior_samples, observation.repeat((1000, 1)))
    trunc_boundary = torch.quantile(log_probs, sample_epsilon)
    # 在计算 trunc_boundary 后添加：
    print("截断边界值:", trunc_boundary.item())
    print("后验样本的最小对数概率:", log_probs.min().item())
    print("后验样本的最大对数概率:", log_probs.max().item())
    print("The type of log_probs is:", type(log_probs))  # 应显示 <class 'torch.Tensor'>
    print("The shape of log_probs is:", log_probs.shape)  # 显示样本数量（如 [1000]）
    print("The device of log_probs is:", log_probs.device)  # 检查是否在预期设备上（cpu/cuda）

    posterior_uniform_hypercube_min = est_posterior_samples.min(axis=0).values  # 后验样本最小值
    posterior_uniform_hypercube_max = est_posterior_samples.max(axis=0).values  # 后验样本最大值
    print("the type of posterior_uniform_hypercube_min: ", type(posterior_uniform_hypercube_min))

    device = est_posterior_samples.device  # 获取后验样本所在的设备
    print(f"est_posterior_samples device: {est_posterior_samples.device}")
    prior_samples = prior(int(1e6)).to(device)
    print(f"prior_samples device: {prior_samples.device}")
    print("the device of est_posterior_samples is:", device)
    prior_uniform_hypercube_min = prior_samples.min(axis=0).values  # 原始先验样本最小值
    prior_uniform_hypercube_max = prior_samples.max(axis=0).values  # 原始先验样本最大值
    print("the type of prior_uniform_hypercube_min: ", type(prior_uniform_hypercube_min))

    hypercube_min = torch.maximum(
        posterior_uniform_hypercube_min.unsqueeze(0),  # 相当于 [None, :]
        prior_uniform_hypercube_min.unsqueeze(0)  # 相当于 [None, :]
    ).max(dim=0).values  # 取最大值

    # 计算 hypercube_max (取两个max值的最小值)
    hypercube_max = torch.minimum(
        posterior_uniform_hypercube_max.unsqueeze(0),  # 相当于 [None, :]
        prior_uniform_hypercube_max.unsqueeze(0)  # 相当于 [None, :]
    ).min(dim=0).values  # 取最小值
    # 在计算 hypercube_min/max 后添加：
    print("超立方体最小值:", hypercube_min)
    print("超立方体最大值:", hypercube_max)
    print("范围有效性检查:", torch.all(hypercube_max > hypercube_min))  # 必须为True
    if logger:
        logger.info(
            f"Posterior uniform hypercube min/max: {posterior_uniform_hypercube_min}, {posterior_uniform_hypercube_max}")
        logger.info(f"Prior uniform hypercube min/max: {prior_uniform_hypercube_min}, {prior_uniform_hypercube_max}")
        logger.info(f"Hypercube min/max: {hypercube_min}, {hypercube_max}")
    else:
        print(f"Posterior min/max: {posterior_uniform_hypercube_min}, {posterior_uniform_hypercube_max}")
        print(f"Prior min/max: {prior_uniform_hypercube_min}, {prior_uniform_hypercube_max}")
        print(f"Hypercube min/max: {hypercube_min}, {hypercube_max}")

    def hypercube_uniform_prior(num_samples):
        D = hypercube_min.shape[0]  # D 和 num_samples 都是 int
        # print("the type of D: ", type(D))
        # print("the type of num_samples: ", type(num_samples))
        # samples = torch.rand((num_samples, D)) * (hypercube_max.cpu() - hypercube_min.cpu()) + hypercube_min.cpu()
        device = hypercube_min.device  # 获取 hypercube_min 的设备
        samples = torch.rand((num_samples, D), device=device) * (hypercube_max - hypercube_min) + hypercube_min
        # print("生成样本的最小值:", samples.min(dim=0).values)
        # print("生成样本的最大值:", samples.max(dim=0).values)
        return samples

    def truncated_prior(num_samples, batch_size):
        # 这里的 num_samples 其实是 simulation_budget_round
        # 从一个经过截断的分布中采样，确保：
        # 1. 采样点必须落在由 hypercube_min 和 hypercube_max 定义的超立方体内
        # 2. 采样点的概率密度必须高于 trunc_boundary
        max_iters = 1000
        # batch_size = 50
        counter = 0
        n_samples_so_far = 0
        samples_out = []
        log_probs = []
        # print("The num_samples is: ", num_samples)
        # print("The shape of trunc_boundary is: ", trunc_boundary.shape)
        # print("The trunc_boundary is: ", trunc_boundary)
        print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        while (n_samples_so_far < num_samples) and (counter < max_iters):
            # -- 生成小批量样本
            current_batch_size = min(batch_size, num_samples - n_samples_so_far)
            samples = hypercube_uniform_prior(current_batch_size)  # 生成小批量样本

            obs_repeated = observation.repeat((current_batch_size, 1))
            log_probs = model.log_prob_batch(samples, obs_repeated)
            # print("生成样本的对数概率范围:", log_probs.min().item(), log_probs.max().item())
            accepted_samples = samples[log_probs > trunc_boundary]  # 对应公式里的指示函数，即I{theta 属于HPR_epsilon}的实现
            samples_out.append(accepted_samples.cpu())
            n_samples_so_far += len(accepted_samples)
            counter += 1

            del samples, obs_repeated, log_probs
            torch.cuda.empty_cache()
            log(f"Iteration {counter}: Generated {n_samples_so_far}/{num_samples} samples, "
                f"accepted {len(accepted_samples)} in this batch", output_file)


        if counter == max_iters:
            assert ValueError(
                "Truncated prior sampling did not converge in the allowed number of iterations - returning error.")
        cpu_samples = [s.cpu().detach().numpy() for s in samples_out]
        array = np.concatenate(cpu_samples)[0:num_samples]
        print("the shape of this array is:", array.shape)
        return np.concatenate(cpu_samples)[0:num_samples]
        # 对应的是每一轮生成的图像的蓝色点, 也是每轮通过 truncated_prior 生成的样本，即 $\bar{p}^r(\theta)$

    return truncated_prior


def log(message, file):
    """打印并写入日志文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 添加时间戳
    formatted_message = f"[{timestamp}] {message}"  # 格式化日志内容
    print(formatted_message)
    with open(file, 'a') as f:
        f.write(formatted_message + '\n')


def c2st_test(reference_samples, posterior_samples):
    reference_samples = reference_samples.to(posterior_samples.device)
    n = min(len(reference_samples), len(posterior_samples))
    c2st_score = c2st(posterior_samples[:n], reference_samples[:n])
    return c2st_score


def draw_and_write(posterior_samples_list, theta, num_rounds,
                   simulation_budget_round, reference_samples, output_dir):
    image_path = os.path.join(output_dir, "posterior_vs_true.png")
    if num_rounds > 1:
        if num_rounds < 4:
            fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        if num_rounds == 5:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        if num_rounds == 10:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()  # 将axes扁平化，以便通过索引访问每个子图

        for ii in range(num_rounds):
            ax = axes[ii]  # 获取当前图的坐标轴对象
            ax.scatter(theta[(ii * simulation_budget_round):((ii + 1) * simulation_budget_round), 0].cpu().numpy(),
                       theta[(ii * simulation_budget_round):((ii + 1) * simulation_budget_round), 1].cpu().numpy(),
                       label=f"round {ii + 1}", s=2)
            # 蓝色的部分，即当前轮次的采样，即用于从参数数组 theta 中按轮次提取当前轮次的所有样本
            if ii < num_rounds:  # 确保有对应的后验样本, 即橙色的部分
                ax.scatter(posterior_samples_list[ii][:, 0].cpu().numpy(),
                           posterior_samples_list[ii][:, 1].cpu().numpy(),
                           label="est posterior", s=2)
            ax.scatter(reference_samples[:, 0],
                       reference_samples[:, 1],
                       label="true post", s=2)  # 绿色的部分，即真实分布
            ax.set_title(f"Round {ii + 1}")
            ax.legend(loc='upper left')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))  # 创建单个图形
        # 绘制第一轮的数据
        ax.scatter(theta[0:simulation_budget_round, 0].cpu().numpy(),
                   theta[0:simulation_budget_round, 1].cpu().numpy(),
                   label="round 1", s=2)
        # 绘制估计的后验
        if len(posterior_samples_list) > 0:
            ax.scatter(posterior_samples_list[0][:, 0].cpu().numpy(),
                       posterior_samples_list[0][:, 1].cpu().numpy(),
                       label="est posterior", s=2)
        # 绘制真实后验
        ax.scatter(reference_samples[:, 0],
                   reference_samples[:, 1],
                   label="true post", s=2)
        ax.set_title("Round 1")
        ax.legend(loc='upper left')

    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存结果图像
    plt.savefig(image_path)
    plt.show()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True,
        help="Base save directory for the evaluation"
    )
    args = parser.parse_args()
    # 完成这步后，args.train_dir 用于访问 --train_dir的值，即 ”./output"
    with open("settings.yaml", "r", encoding="utf-8") as f:  # 强制使用 UTF-8
        settings = yaml.safe_load(f)  # 返回一个字典，其内容是 settings.yaml 的内容

    benchmark = settings["task"]["name"]
    num_rounds = settings["task"]["num_rounds"]
    simulation_budget = settings["task"]["num_train_samples"]
    device = settings["training"].get("device", "cuda")
    sample_epsilon = float(settings["sampling"]["epsilon"])
    print("the sample_epsilon is: ", sample_epsilon)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('output', benchmark,
                              f"simulation_budget_{simulation_budget}",
                              f"figure_test_num_rounds_{num_rounds}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{timestamp}.txt")

    log(f"The benchmark is: {benchmark}", output_file)
    log(f"The simulation_budget is: {simulation_budget}", output_file)
    log(f"The num_rounds is: {num_rounds}", output_file)
    log(f"The truncated epsilon is: {sample_epsilon}", output_file)
    logger = setup_logger(output_file)
    logger.info("Starting run function...")

    task = sbibm.get_task(benchmark)
    simulator, prior = get_sim_and_prior_from_sbibm(task)
    simulation_budget_round = int(simulation_budget / num_rounds)

    rr = 0
    obs = 1
    batch_size = 500
    posterior_samples_list = []
    c2st_value = []
    reference_samples = task.get_reference_posterior_samples(num_observation=obs)
    # print("the shape of reference_samples is:",reference_samples.shape)
    ## Output: [simulation_budget, dim]
    num_samples = len(reference_samples)
    st = time.time()
    for rr in range(num_rounds):
        logger.info(f"Starting round {rr + 1}/{num_rounds}")
        if rr == 0:
            # dataset = generate_dataset(benchmark, num_rounds, simulation_budget)
            theta = prior(simulation_budget_round)
            x = simulator(theta)
            parameter_ds = torch.tensor(theta, dtype=torch.float)
            data_ds = torch.tensor(x, dtype=torch.float)

        else:
            # reference_samples = task.get_reference_posterior_samples(num_observation=obs)
            # print("the shape of reference_samples is:",reference_samples.shape)
            ## Output: [simulation_budget, dim]
            # num_samples = len(reference_samples)
            observation = dataset.standardize(task.get_observation(num_observation=obs), label="x")
            truncated_prior = get_truncated_prior(sample_epsilon, batch_size, num_samples,
                                                  observation, prior, model=model, logger=logger)
            trunc_prior_samps = truncated_prior(simulation_budget_round, batch_size)
            trunc_prior_samps_tensor = torch.from_numpy(trunc_prior_samps)
            parameter_ds = torch.cat([parameter_ds, trunc_prior_samps_tensor], dim=0)
            data_ds = torch.cat([data_ds, simulator(trunc_prior_samps_tensor)], dim=0)
            # 删除本轮不再使用的变量
            # del train_loader, test_loader, model, est_posterior_samples
            # 强制释放 GPU 缓存
            # torch.cuda.empty_cache()

        dataset = SbiDataset(parameter_ds, data_ds)
        train_loader, test_loader = build_train_and_test_loaders(
            dataset,
            settings["training"]["train_fraction"],  # 训练集占比
            settings["training"]["batch_size"],  # 每个批次的样本数
            settings["training"]["num_workers"],  # 数据加载的并行进程数
        )

        model = train_model(
            args.train_dir,
            settings=settings,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        observation = dataset.standardize(task.get_observation(num_observation=obs), label="x")
        est_posterior_samples = model.sample_batch(observation.repeat((num_samples, 1)))
        est_posterior_samples = dataset.standardize(est_posterior_samples, label="theta", inverse=True)
        # c2st_score = c2st_test(reference_samples, est_posterior_samples)
        # print(f"The C2ST in round {rr + 1} is: ", c2st_score)
        # log(f"The C2ST in round {rr + 1} is: {c2st_score}", output_file)
        # c2st_value.append(c2st_score)
        posterior_samples_list.append(est_posterior_samples)

        if rr == num_rounds - 1:
            # parameter_ds 对应蓝色的部分
            # posterior_samples_list 对应橙色的部分
            # reference_samples 对应绿色的部分
            # for i in range(num_rounds):
            #     print(f"c2st in round {i + 1} is:", c2st_value[i])
            #     log(f"c2st in round {i + 1} is: {c2st_value[i]} ", output_file)
            draw_and_write(posterior_samples_list, parameter_ds, num_rounds,
                           simulation_budget_round, reference_samples, output_dir)

    log(f"The run took {time.time() - st} seconds\n", output_file)