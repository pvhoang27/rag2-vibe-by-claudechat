"""
scripts/run_evaluation.py
--------------------------
CLI script to run RAGAS evaluation with pre-built sample QA pairs.
Usage: python scripts/run_evaluation.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Sample QA pairs (question + ground truth) ──────────────────────────────────
EVAL_SAMPLES = [
    {
        "question": "AI là gì và được phát minh vào năm nào?",
        "ground_truth": "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính tập trung vào xây dựng hệ thống thông minh. AI được khai sinh vào năm 1956 tại Hội nghị Dartmouth."
    },
    {
        "question": "Machine Learning có những loại nào?",
        "ground_truth": "Machine Learning có 3 loại chính: Supervised Learning (học có giám sát), Unsupervised Learning (học không giám sát), và Reinforcement Learning (học tăng cường)."
    },
    {
        "question": "RAG là gì và tại sao cần dùng RAG thay vì LLM thuần túy?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) kết hợp khả năng sinh văn bản của LLM với truy xuất thông tin từ cơ sở dữ liệu. Cần RAG vì LLM thuần túy bị giới hạn bởi knowledge cutoff, có thể hallucination và không có kiến thức về tài liệu nội bộ."
    },
    {
        "question": "RAGAS đánh giá RAG bằng những metrics nào?",
        "ground_truth": "RAGAS đánh giá RAG với 4 metrics: Faithfulness (câu trả lời dựa trên context), Answer Relevancy (liên quan câu hỏi), Context Recall (context bao phủ thông tin cần thiết), Context Precision (context chính xác, ít nhiễu)."
    },
    {
        "question": "ChromaDB là gì và ưu điểm của nó?",
        "ground_truth": "ChromaDB là vector database open-source, nhẹ, phù hợp cho prototype. Nó được dùng để lưu trữ embedding vectors trong hệ thống RAG."
    },
    {
        "question": "LangChain dùng để làm gì?",
        "ground_truth": "LangChain là framework xây dựng ứng dụng với LLM, hỗ trợ RAG, Agents, Chains và tích hợp với hầu hết LLM và vector stores."
    },
    {
        "question": "Supervised Learning khác Unsupervised Learning như thế nào?",
        "ground_truth": "Supervised Learning được huấn luyện trên dữ liệu đã gán nhãn để phân loại hoặc dự đoán, còn Unsupervised Learning tự tìm kiếm cấu trúc ẩn trong dữ liệu không có nhãn."
    },
    {
        "question": "Python có những thư viện AI nào phổ biến?",
        "ground_truth": "Các thư viện AI Python phổ biến gồm: NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow/Keras, LangChain, và HuggingFace Transformers."
    },
]


async def main():
    import httpx

    BASE_URL = "http://localhost:8000"

    console.print(Panel.fit(
        "[bold cyan]RAG Chatbot — RAGAS Evaluation Runner[/bold cyan]\n"
        f"Evaluating {len(EVAL_SAMPLES)} QA pairs",
        border_style="cyan"
    ))

    # Check server health
    console.print("\n[yellow]1. Checking server health…[/yellow]")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BASE_URL}/health")
            r.raise_for_status()
        console.print("   [green]✓ Server is running[/green]")
    except Exception as e:
        console.print(f"   [red]✗ Server not reachable: {e}[/red]")
        console.print("   [dim]Please start the server with: uv run uvicorn app.main:app --reload[/dim]")
        sys.exit(1)

    # Check collection has documents
    console.print("\n[yellow]2. Checking vector collection…[/yellow]")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}/chat/collection")
        info = r.json()
        doc_count = info.get("document_count", 0)
        console.print(f"   [green]✓ Collection '{info['name']}' has {doc_count} chunks[/green]")
        if doc_count == 0:
            console.print("   [red]✗ No documents found! Please ingest data first.[/red]")
            console.print("   [dim]Run: python scripts/generate_sample_data.py[/dim]")
            console.print("   [dim]Then: POST /ingest/directory via API[/dim]")
            sys.exit(1)

    # Run evaluation
    console.print(f"\n[yellow]3. Running RAGAS evaluation on {len(EVAL_SAMPLES)} samples…[/yellow]")
    console.print("   [dim]This may take several minutes depending on your machine…[/dim]\n")

    payload = {"samples": EVAL_SAMPLES}
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{BASE_URL}/eval/run", json=payload)
        response.raise_for_status()
        report = response.json()

    # Display results table
    table = Table(
        title="\n📊 RAGAS Evaluation Report",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=22)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Grade", justify="center", width=10)
    table.add_column("Description", style="dim")

    for metric in report["metrics"]:
        score = metric["score"]
        if score >= 0.8:
            grade, color = "Excellent", "green"
        elif score >= 0.6:
            grade, color = "Good", "yellow"
        elif score >= 0.4:
            grade, color = "Fair", "orange3"
        else:
            grade, color = "Poor", "red"

        table.add_row(
            metric["name"],
            f"[{color}]{score:.4f}[/{color}]",
            f"[{color}]{grade}[/{color}]",
            metric["description"],
        )

    table.add_section()
    overall = report["overall_score"]
    o_color = "green" if overall >= 0.7 else "yellow" if overall >= 0.5 else "red"
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold {o_color}]{overall:.4f}[/bold {o_color}]",
        "",
        f"Model: {report['model']} | Samples: {report['sample_count']}",
    )

    console.print(table)

    if report.get("output_path"):
        console.print(f"\n[dim]Full report saved to: {report['output_path']}[/dim]")

    console.print("\n[bold green]✅ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
