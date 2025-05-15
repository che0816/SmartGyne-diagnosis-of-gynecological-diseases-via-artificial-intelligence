# README

## Description
This script is designed to generate a knowledge graph based on a feature table and a technical word dictionary. It takes input files and parameters through command-line arguments and produces the knowledge graph in a specified output directory.

## Usage
To run this script, you need to provide the following command-line arguments:

- `--feature_table_path`: The path to the feature table file. This file should contain the necessary data for generating the knowledge graph.
- `--technical_word_dict_dir`: The path to the technical word dictionary file. This file should contain the technical terms and their relationships.
- `--output_dir`: The directory where the generated knowledge graph will be saved.
- `--thr`: The threshold value for the algorithm (default: 0.05).

### Example Command
```bash
python script_name.py --feature_table_path "path/to/feature_table" --technical_word_dict_dir "path/to/technical_word_dict.json" --output_dir "path/to/output_directory" --thr 0.05
```

## Dependencies
- `argparse`， `scipy`, `statsmodels`, `tqdm`

## Notes
- Ensure that the input files (`feature_table` and `technical_word_dict`) are in the correct format and contain the necessary data.
- The output directory should have write permissions.
- The threshold value (`--thr`) can be adjusted based on the specific requirements of the algorithm.

## Example Input Files
- **Feature Table**: A feature path must contains `disease_list.json` and `feature_data_seg_standard.json`.

  - disease_list.json:

  ```bash
  ["产后身痛", "腹壁多发性内异症术后", "人流后复诊", ...]
  ```

  - feature_data_seg_standard.json

    ```bash
    [
     {"诊断名称": 2435, "主诉": [], "处理意见": ["云南白药胶囊", "宫颈", "活检标本病理诊断", "人乳头瘤病毒", "超薄细胞检测", "高危型人乳头瘤病毒基因分型检测"], "既往史": [], "查体": [], "现病史": ["新型冠状病毒核酸检测"], "辅助检查": ["新型冠状病毒核酸检测"]}, 
     {"诊断名称": 648, "主诉": ["月经不规律"], "处理意见": ["HGB", "琥珀酸亚铁片", "子宫双附件超声检查", "速力菲", "地屈孕酮片", "地屈孕酮片", "全血细胞分析"], "既往史": [], "查体": ["POP", "肌力"], "现病史": ["内膜厚", "脱垂"], "辅助检查": []},
     ...]
    ```

- **Technical Word Dictionary**: A JSON file containing technical terms and their relationships.

  ```bash
  {"淋巴结彩超检查": "检查项目", "宫颈鳞癌IB1": "疾病", ...}
  ```

## Example Output
- The generated knowledge graph will be saved in the specified output directory in an Excel file format (e.g., `knowledge_graph.xlsx`).

