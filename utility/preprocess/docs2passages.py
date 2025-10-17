# 文件名: utility/preprocess/docs2passages.py

import os
import random
from multiprocessing import Pool
from argparse import ArgumentParser

# 导入 ColBERT 的工具函数
from colbert.utils.utils import print_message

# 定义支持的输入文件格式
Format1 = 'docid,text'  # 例如: MS MARCO Passages
Format2 = 'docid,text,title'   # 例如: DPR Wikipedia
Format3 = 'docid,url,title,text'  # 例如: MS MARCO Documents


def process_page(inp):
    """
    处理单个文档（页面），将其切分为多个段落。
    这个函数被设计为可以由多进程并行调用。

    切分策略：
    - 使用滑动窗口将文档切分为 `nwords` 长度的段落。
    - 窗口的步长是 `nwords - overlap`。
    - 为了避免最后一个段落过短，它会从文档的开头“回绕”（wrap-around）补充内容，
      确保所有生成的段落都有足够的长度。这类似于 DPR 的预处理方法。
    """
    # 解析输入参数
    (nwords, overlap, tokenizer), (line_idx, docid, title, url, content) = inp

    # 根据是否使用 wordpiece tokenizer 来选择不同的分词方式
    if tokenizer is None:
        words = content.split()
    else:
        words = tokenizer.tokenize(content)

    # 如果文档本身比目标段落长度还短，则直接使用；否则，复制一份自身以支持“回绕”
    words_to_slide = (words + words) if len(words) > nwords else words
    
    # 使用列表推导式高效地切分段落
    passages = [
        words_to_slide[offset : offset + nwords]
        for offset in range(0, len(words), nwords - overlap)
    ]

    # 将切分后的 token 列表重新组合成字符串
    if tokenizer is None:
        passages = [' '.join(psg) for psg in passages]
    else:
        # 对于 wordpiece，需要处理 '##' 前缀
        passages = [' '.join(psg).replace(' ##', '') for psg in passages]

    # 返回处理结果
    return (docid, title, url, passages)


def main(args):
    """主函数，协调整个文档到段落的转换过程。"""
    random.seed(12345)
    print_message("#> 开始处理...")

    # 根据分词方式确定输出文件名后缀
    suffix_char = 't' if args.use_wordpiece else 'w'
    output_path = f'{args.input}.{suffix_char}{args.nwords}_{args.overlap}'
    assert not os.path.exists(output_path), f"输出文件 {output_path} 已存在！"

    # --- 1. 读取原始文档集合 ---
    raw_collection = []
    print_message(f"#> 正在从 {args.input} 加载文档...")
    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            if (line_idx + 1) % 100000 == 0:
                print(f"已读取 {line_idx + 1} 行...", end='\r')
            
            try:
                parts = line.strip().split('\t')
                docid, url, title, doc = None, None, None, None
                # 根据指定的格式解析每一行
                if args.format == Format1:
                    docid, doc = parts
                elif args.format == Format2:
                    docid, doc, title = parts
                elif args.format == Format3:
                    docid, url, title, doc = parts
                
                raw_collection.append((line_idx, docid, title, url, doc))
            except (ValueError, IndexError):
                # 跳过格式不正确的行
                continue
    print(f"\n#> 加载了 {len(raw_collection)} 篇文档。\n")

    # --- 2. 并行处理 ---
    print_message("#> 开始并行切分段落...")
    pool = Pool(args.nthreads)
    
    tokenizer = None
    if args.use_wordpiece:
        # 如果使用 wordpiece，动态导入并初始化 BertTokenizerFast
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 准备传递给 `process_page` 函数的参数
    process_params = [(args.nwords, args.overlap, tokenizer)] * len(raw_collection)
    
    # 使用 pool.map 并行执行切分任务
    processed_collection = pool.map(process_page, zip(process_params, raw_collection))

    # --- 3. 写入输出文件 ---
    print_message(f"#> 正在将结果写入 {output_path} ...")
    with open(output_path, 'w') as f:
        passage_idx = 1
        # 写入表头
        if args.format == Format1:
            f.write('id\ttext\n')
        elif args.format == Format2:
            f.write('id\ttext\ttitle\n')
        elif args.format == Format3:
            f.write('id\ttext\ttitle\tdocid\n')
        
        # 遍历处理后的文档，将每个段落写入新的一行
        for docid, title, url, passages in processed_collection:
            for passage_text in passages:
                if args.format == Format1:
                    f.write(f'{passage_idx}\t{passage_text}\n')
                elif args.format == Format2:
                    f.write(f'{passage_idx}\t{passage_text}\t{title}\n')
                elif args.format == Format3:
                    f.write(f'{passage_idx}\t{passage_text}\t{title}\t{docid}\n')
                passage_idx += 1
    
    print_message("#> 完成！")


if __name__ == "__main__":
    parser = ArgumentParser(description="将长文档集合切分为固定长度的段落。")
    parser.add_argument('--input', dest='input', required=True, help="输入的文档集合文件路径 (.tsv 格式)")
    parser.add_argument('--format', dest='format', required=True, choices=[Format1, Format2, Format3], help="输入文件的格式")
    parser.add_argument('--nwords', dest='nwords', default=180, type=int, help="每个段落的目标词数（或 token 数）")
    parser.add_argument('--overlap', dest='overlap', default=90, type=int, help="切分时滑动窗口的重叠词数")
    parser.add_argument('--use-wordpiece', dest='use_wordpiece', action='store_true', help="使用 BERT 的 WordPiece 分词器代替按空格分词")
    parser.add_argument('--nthreads', dest='nthreads', default=16, type=int, help="用于并行处理的进程数")
    args = parser.parse_args()
    
    assert args.nwords > args.overlap, "段落长度必须大于重叠长度"
    
    main(args)