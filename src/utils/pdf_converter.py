import os
import logging
from typing import Optional
from llama_index.core import SimpleDirectoryReader

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToMarkdownConverter:
    """
    PDF 转换器：将 data/raw 中的 PDF 转换为 data/processed/markdown 中的 .md 文件。
    支持 LlamaParse (结构化解析) 和本地解析模式。
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")

    def convert_file(self, file_path: str, use_llama_parse: bool = False):
        """转换单个文件"""
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        output_file = os.path.join(self.output_dir, f"{base_name}.md")
        
        logger.info(f"开始转换: {file_name} ...")
        
        try:
            if use_llama_parse:
                # 推荐方式：需要安装 llama-parse 并配置 LLAMA_CLOUD_API_KEY
                # pip install llama-parse
                from llama_parse import LlamaParse
                parser = LlamaParse(result_type="markdown", verbose=True)
                documents = parser.load_data(file_path)
            else:
                # 本地方式：使用 SimpleDirectoryReader
                # 如果 pdf 结构简单，这种方式也足够
                documents = SimpleDirectoryReader(
                    input_files=[file_path]
                ).load_data()

            # 将所有页面的内容合并成一个 Markdown
            full_content = "\n\n".join([doc.text for doc in documents])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            logger.info(f"转换成功: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"转换失败 {file_name}: {e}")
            return None

    def convert_all(self, input_dir: str, use_llama_parse: bool = False):
        """批量转换目录下所有 PDF"""
        converted_count = 0
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    result = self.convert_file(pdf_path, use_llama_parse)
                    if result:
                        converted_count += 1
        
        logger.info(f"\n--- 批量转换完成: 处理了 {converted_count} 个文件 ---")

if __name__ == "__main__":
    # 使用示例
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_path = os.path.join(project_root, "data", "raw", "books")
    md_output_path = os.path.join(project_root, "data", "processed", "markdown")
    
    converter = PDFToMarkdownConverter(md_output_path)
    
    # 默认使用本地模式，如果有 API Key，可以将 use_llama_parse 设为 True
    # 注意：使用 LlamaParse 需先在 .env 或环境变量中设置 LLAMA_CLOUD_API_KEY
    converter.convert_all(raw_path, use_llama_parse=False)
