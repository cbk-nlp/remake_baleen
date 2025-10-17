# 文件名: utility/rankings/tune.py

import os
import ujson
from argparse import ArgumentParser

from colbert.utils.utils import print_message, create_directory
from utility.utils.save_metadata import save_metadata


def main(args):
    """主函数，负责读取指标、找到最佳模型并保存结果。"""
    all_metrics = {}
    scores = {}

    # --- 1. 读取所有指标文件 ---
    print_message(f"#> 正在从 {len(args.paths)} 个指标文件中读取 '{args.metric}' 指标...")
    for path in args.paths:
        with open(path) as f:
            metric_data = ujson.load(f)
            all_metrics[path] = metric_data
            
            # 嵌套地访问指定的指标，例如 ['success', '20'] -> metric_data['success']['20']
            score = metric_data
            for key in args.metric:
                score = score[key]
            
            assert isinstance(score, float)
            scores[path] = score
    
    # --- 2. 找到分数最高的检查点 ---
    # 找到得分最高的指标文件路径
    best_metric_path = max(scores, key=scores.get)
    best_score = scores[best_metric_path]
    print_message(f"#> 最佳表现来自: {best_metric_path}")
    print_message(f"#> 最佳分数为: {best_score:.4f}")

    # 从最佳指标文件的路径推断出对应的检查点编号和日志文件路径
    # 假设指标文件路径类似于: .../checkpoints/colbert-CKPT_NUM/ranking.tsv.annotated.metrics
    parts = best_metric_path.split('/')
    try:
        # 这是一个启发式的路径解析，可能需要根据实际目录结构调整
        ckpt_num_str = [p for p in parts if p.startswith('colbert-') and not p.endswith('.dnn')][-1]
        ckpt_num = int(ckpt_num_str.split('-')[-1])
        
        # 假设 args.json 位于 logs/ 目录下
        args_json_path = os.path.join(os.path.dirname(best_metric_path), '..', 'logs', 'args.json')
        assert os.path.exists(args_json_path)
    except (IndexError, AssertionError, ValueError):
        print("[错误] 无法从路径中自动推断检查点编号和日志路径。请检查文件结构。")
        return

    # --- 3. 从日志中找到原始的检查点路径并保存 ---
    with open(args_json_path) as f:
        logs = ujson.load(f)
        # 原始检查点路径通常保存在 'checkpoint' 字段中
        best_checkpoint_path = logs['checkpoint']
        
        # 验证检查点路径是否与我们推断的编号匹配
        assert best_checkpoint_path.endswith(f'{ckpt_num}.dnn') or f'-{ckpt_num}/' in best_checkpoint_path

    print_message(f"#> 对应的最佳检查点路径是: {best_checkpoint_path}")
    
    # 将找到的最佳检查点路径写入输出文件
    with open(args.output, 'w') as f:
        f.write(best_checkpoint_path)

    # 保存本次调优过程的元数据
    args.Scores = scores
    args.AllMetrics = all_metrics
    save_metadata(f'{args.output}.meta', args)

    print(f"\n#> 已将最佳检查点路径写入: {args.output}")
    print("#> 完成。")


if __name__ == "__main__":
    parser = ArgumentParser(description='从多个评估指标文件中自动选择最佳模型检查点。')
    parser.add_argument('--metric', dest='metric', required=True, type=str, help="用于调优的目标指标，用点号分隔，例如 'success.20'")
    parser.add_argument('--paths', dest='paths', required=True, nargs='+', type=str, help="一个或多个评估指标文件 (.json) 的路径")
    parser.add_argument('--output', dest='output', required=True, type=str, help="用于保存最佳检查点路径的输出文件")
    args = parser.parse_args()

    # 将点号分隔的指标字符串转换为列表
    args.metric = args.metric.split('.')

    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"
    create_directory(os.path.dirname(args.output))

    main(args)