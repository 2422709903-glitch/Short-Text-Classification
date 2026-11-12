"""只对训练集少数类别做可复现的随机过采样，不改动 dev/test。"""

import random
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / 'data'
SOURCE_PATH = DATA_DIR / 'train.txt'
OUTPUT_PATH = DATA_DIR / 'train_balanced.txt'
TARGET_PER_CLASS = 1000
RANDOM_SEED = 42


def main():
    class_rows = defaultdict(list)
    with SOURCE_PATH.open(encoding='utf-8') as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip('\r\n')
            if not line:
                continue
            fields = line.split('\t', 1)
            if len(fields) != 2 or not fields[0].isdigit() or not fields[1].strip():
                raise ValueError(f'{SOURCE_PATH}:{line_number}: 需要非空的 label<TAB>text 数据')
            class_rows[int(fields[0])].append(line)

    rng = random.Random(RANDOM_SEED)
    output_rows = [row for rows in class_rows.values() for row in rows]
    added = 0
    for class_id, rows in class_rows.items():
        missing = max(0, TARGET_PER_CLASS - len(rows))
        if missing:
            output_rows.extend(rng.choices(rows, k=missing))
            added += missing

    rng.shuffle(output_rows)
    with OUTPUT_PATH.open('w', encoding='utf-8', newline='\n') as target:
        target.write('\n'.join(output_rows))
        target.write('\n')

    print(f'原训练集：{sum(map(len, class_rows.values()))} 条')
    print(f'补充训练样本：{added} 条；输出：{len(output_rows)} 条')
    print(f'少数类最低样本数：{min(max(len(rows), TARGET_PER_CLASS) for rows in class_rows.values())}')
    print(f'写入文件：{OUTPUT_PATH}')


if __name__ == '__main__':
    main()

