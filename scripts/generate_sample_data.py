"""
scripts/generate_sample_data.py
--------------------------------
Generates sample Vietnamese text documents for testing the RAG pipeline.
Run: python scripts/generate_sample_data.py
"""

from pathlib import Path

SAMPLE_DOCS = {
    "gioi_thieu_ai.txt": """
Trí Tuệ Nhân Tạo (AI) - Tổng Quan

Trí tuệ nhân tạo (Artificial Intelligence - AI) là lĩnh vực khoa học máy tính tập trung vào
việc xây dựng các hệ thống có khả năng thực hiện các tác vụ thường đòi hỏi trí thông minh
của con người.

Lịch sử phát triển AI:
AI được khai sinh vào năm 1956 tại Hội nghị Dartmouth, nơi John McCarthy lần đầu tiên
đặt ra thuật ngữ "trí tuệ nhân tạo". Từ đó đến nay, AI đã trải qua nhiều giai đoạn thăng trầm.

Giai đoạn 1956-1974 được gọi là "Mùa xuân AI" đầu tiên, khi các nhà nghiên cứu rất lạc quan
về tiềm năng của AI. Tuy nhiên, do hạn chế về phần cứng và dữ liệu, AI rơi vào "Mùa đông AI"
lần đầu tiên vào những năm 1974-1980.

Các nhánh chính của AI:
1. Machine Learning (Học máy): Cho phép máy tính học từ dữ liệu mà không cần lập trình
   rõ ràng. Bao gồm supervised learning, unsupervised learning và reinforcement learning.

2. Deep Learning (Học sâu): Sử dụng mạng nơ-ron nhân tạo nhiều lớp để xử lý dữ liệu
   phức tạp. Đây là nền tảng của nhiều ứng dụng AI hiện đại.

3. Natural Language Processing (NLP): Cho phép máy tính hiểu và xử lý ngôn ngữ tự nhiên
   của con người.

4. Computer Vision: Cho phép máy tính "nhìn" và hiểu hình ảnh, video.

Ứng dụng AI trong thực tế:
- Y tế: Chẩn đoán bệnh qua hình ảnh, phát hiện ung thư sớm
- Tài chính: Phát hiện gian lận, dự đoán thị trường
- Giao thông: Xe tự lái, tối ưu hóa giao thông
- Giáo dục: Cá nhân hóa lộ trình học tập
- Sản xuất: Robot công nghiệp, kiểm soát chất lượng
""",

    "machine_learning.txt": """
Machine Learning - Học Máy

Machine Learning (ML) là một nhánh của AI cho phép hệ thống học hỏi và cải thiện từ
kinh nghiệm mà không cần lập trình rõ ràng.

Các loại Machine Learning:

1. Supervised Learning (Học có giám sát):
   Mô hình được huấn luyện trên tập dữ liệu đã được gán nhãn. Ví dụ:
   - Phân loại email spam/không spam
   - Dự đoán giá nhà
   - Nhận diện khuôn mặt
   
   Thuật toán phổ biến: Linear Regression, Logistic Regression, Decision Tree,
   Random Forest, Support Vector Machine (SVM), Neural Networks.

2. Unsupervised Learning (Học không giám sát):
   Mô hình tự tìm kiếm cấu trúc ẩn trong dữ liệu không có nhãn. Ví dụ:
   - Phân nhóm khách hàng (Customer Segmentation)
   - Phát hiện bất thường (Anomaly Detection)
   - Giảm chiều dữ liệu
   
   Thuật toán phổ biến: K-Means Clustering, DBSCAN, PCA, Autoencoders.

3. Reinforcement Learning (Học tăng cường):
   Agent học cách hành động trong môi trường để tối đa hóa phần thưởng. Ví dụ:
   - AlphaGo (chơi cờ vây)
   - Điều khiển robot
   - Tối ưu hóa quảng cáo

Quy trình xây dựng mô hình ML:
1. Thu thập và làm sạch dữ liệu (Data Collection & Cleaning)
2. Phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis)
3. Feature Engineering (Kỹ thuật đặc trưng)
4. Chọn và huấn luyện mô hình (Model Selection & Training)
5. Đánh giá mô hình (Model Evaluation)
6. Triển khai (Deployment)
7. Giám sát và cập nhật (Monitoring & Maintenance)

Các metrics đánh giá mô hình:
- Accuracy, Precision, Recall, F1-Score (cho bài toán phân loại)
- MAE, RMSE, R² (cho bài toán hồi quy)
- AUC-ROC (đánh giá tổng thể classifier)
""",

    "rag_va_llm.txt": """
RAG (Retrieval-Augmented Generation) và Large Language Models

Large Language Models (LLMs):
LLM là các mô hình ngôn ngữ lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.
Các LLM nổi tiếng bao gồm GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google),
LLaMA (Meta), Mistral.

Hạn chế của LLM thuần túy:
- Knowledge cutoff: Kiến thức bị giới hạn đến thời điểm huấn luyện
- Hallucination: Có thể bịa đặt thông tin không chính xác
- Không có kiến thức về tài liệu nội bộ của tổ chức
- Không thể cập nhật thông tin theo thời gian thực

RAG (Retrieval-Augmented Generation):
RAG là kiến trúc kết hợp khả năng sinh văn bản của LLM với việc truy xuất thông tin
từ cơ sở dữ liệu vector. Điều này giúp LLM trả lời dựa trên tài liệu thực tế, giảm
thiểu hallucination.

Kiến trúc RAG cơ bản:
1. Indexing Phase (Giai đoạn lập chỉ mục):
   - Load tài liệu từ nhiều nguồn (PDF, TXT, Web...)
   - Chia tài liệu thành các chunk nhỏ
   - Tạo embedding vector cho mỗi chunk
   - Lưu vào vector database (ChromaDB, Pinecone, Weaviate...)

2. Retrieval Phase (Giai đoạn truy xuất):
   - Người dùng đặt câu hỏi
   - Câu hỏi được chuyển thành embedding vector
   - Tìm kiếm similarity trong vector database
   - Lấy top-k chunks liên quan nhất

3. Generation Phase (Giai đoạn sinh):
   - Kết hợp câu hỏi + context đã retrieve
   - Đưa vào LLM để sinh câu trả lời
   - Trả về câu trả lời kèm nguồn tham khảo

Đánh giá chất lượng RAG với RAGAS:
RAGAS là framework chuyên dụng để đánh giá hệ thống RAG với các metrics:
- Faithfulness: Câu trả lời có dựa trên context không?
- Answer Relevancy: Câu trả lời có liên quan câu hỏi không?
- Context Recall: Context có bao phủ đủ thông tin cần thiết không?
- Context Precision: Context retrieve có chính xác không, ít nhiễu không?

Vector Databases phổ biến:
- ChromaDB: Open-source, nhẹ, phù hợp prototype
- Pinecone: Cloud-based, scalable, production-ready
- Weaviate: Open-source, hỗ trợ hybrid search
- Qdrant: Hiệu năng cao, hỗ trợ filtering phức tạp
- FAISS: Thư viện Facebook, tối ưu tốc độ tìm kiếm
""",

    "python_cho_ai.txt": """
Python cho AI và Data Science

Python là ngôn ngữ lập trình phổ biến nhất trong lĩnh vực AI và Data Science nhờ
cú pháp đơn giản, hệ sinh thái thư viện phong phú và cộng đồng lớn.

Thư viện AI/ML quan trọng:

1. NumPy:
   - Tính toán số học với mảng đa chiều
   - Nền tảng của hầu hết thư viện AI khác
   - Ví dụ: np.array(), np.dot(), np.reshape()

2. Pandas:
   - Xử lý và phân tích dữ liệu dạng bảng
   - DataFrame là cấu trúc dữ liệu chính
   - Ví dụ: pd.read_csv(), df.groupby(), df.merge()

3. Scikit-learn:
   - Thư viện ML toàn diện nhất cho Python
   - Hỗ trợ hầu hết thuật toán ML cổ điển
   - API nhất quán: fit(), predict(), transform()

4. PyTorch:
   - Framework Deep Learning phổ biến nhất trong nghiên cứu
   - Dynamic computation graph
   - Được dùng trong hầu hết LLM hiện đại

5. TensorFlow/Keras:
   - Framework Deep Learning của Google
   - Production-ready, hỗ trợ mobile và web deployment

6. LangChain:
   - Framework xây dựng ứng dụng với LLM
   - Hỗ trợ RAG, Agents, Chains
   - Tích hợp với hầu hết LLM và vector stores

7. HuggingFace Transformers:
   - Hub của hàng nghìn pre-trained models
   - Dễ dàng fine-tuning và inference

Môi trường phát triển:
- Jupyter Notebook/Lab: Ideal cho EDA và prototyping
- VS Code: IDE phổ biến với extensions mạnh mẽ
- Google Colab: Free GPU/TPU trên cloud
- uv: Package manager Python thế hệ mới, nhanh hơn pip nhiều lần

Best Practices khi code AI:
1. Luôn set random seed để reproducibility
2. Version control dữ liệu và model (DVC, MLflow)
3. Logging đầy đủ quá trình training
4. Tách biệt config ra file riêng (.env, yaml)
5. Viết unit test cho preprocessing pipeline
6. Document code và model card rõ ràng
"""
}


def generate_sample_data(output_dir: str = "./data/raw") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        file_path = out_path / filename
        file_path.write_text(content.strip(), encoding="utf-8")
        print(f"  ✓ Created: {file_path}")

    print(f"\n✅ Generated {len(SAMPLE_DOCS)} sample documents in '{output_dir}'")


if __name__ == "__main__":
    print("📄 Generating sample data …\n")
    generate_sample_data()
