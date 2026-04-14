"""
scripts/run_evaluation.py
--------------------------
CLI script to run RAGAS evaluation with pre-built sample QA pairs.
Usage: python scripts/run_evaluation.py
"""

import asyncio
import argparse
import sys
import time
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

DEFAULT_SAMPLE_COUNT = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against local API.")
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of QA samples to evaluate (default: {DEFAULT_SAMPLE_COUNT}, max: {len(EVAL_SAMPLES)})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="API base URL (default: http://127.0.0.1:8000)",
    )
    return parser.parse_args()


async def _progress_heartbeat(stop_event: asyncio.Event, interval: int = 10) -> None:
    """Print periodic progress updates while waiting for a long-running request."""
    start = time.perf_counter()
    tick = 0
    spinner = ["|", "/", "-", "\\"]
    while not stop_event.is_set():
        elapsed = int(time.perf_counter() - start)
        mm, ss = divmod(elapsed, 60)
        icon = spinner[tick % len(spinner)]
        console.print(
            f"   [dim]{icon} Processing... elapsed {mm:02d}:{ss:02d} (waiting for /eval/run)[/dim]"
        )
        tick += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def _wait_for_server(base_url: str, attempts: int = 5, timeout_sec: int = 5) -> bool:
    """Retry health checks to survive brief server reload windows."""
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                r = await client.get(f"{base_url}/health")
                r.raise_for_status()
            return True
        except Exception as exc:
            if attempt == attempts:
                detail = str(exc).strip() or repr(exc)
                console.print(f"   [red]✗ Server not reachable: {detail}[/red]")
                return False
            console.print(
                f"   [dim]Health check attempt {attempt}/{attempts} failed, retrying in 2s...[/dim]"
            )
            await asyncio.sleep(2)
    return False


async def main():
    import httpx

    args = _parse_args()
    requested_samples = max(1, min(args.samples, len(EVAL_SAMPLES)))
    selected_samples = EVAL_SAMPLES[:requested_samples]

    BASE_URL = args.base_url.rstrip("/")

    console.print(Panel.fit(
        "[bold cyan]RAG Chatbot — RAGAS Evaluation Runner[/bold cyan]\n"
        f"Evaluating {len(selected_samples)}/{len(EVAL_SAMPLES)} QA pairs",
        border_style="cyan"
    ))

    # Check server health
    console.print("\n[yellow]1. Checking server health…[/yellow]")
    if await _wait_for_server(BASE_URL, attempts=5, timeout_sec=5):
        console.print("   [green]✓ Server is running[/green]")
    else:
        console.print("   [dim]Please start the server with: uv run uvicorn app.main:app --reload[/dim]")
        console.print(f"   [dim]Current BASE_URL: {BASE_URL}[/dim]")
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
    console.print(f"\n[yellow]3. Running RAGAS evaluation on {len(selected_samples)} samples…[/yellow]")
    console.print("   [dim]This may take several minutes depending on your machine…[/dim]\n")

    payload = {"samples": selected_samples}
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(_progress_heartbeat(stop_event, interval=10))

    report = None
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(f"{BASE_URL}/eval/run", json=payload)
            response.raise_for_status()
            report = response.json()
    except httpx.ReadTimeout:
        console.print("   [red]✗ Request timeout after 10 minutes (600s).[/red]")
        console.print("   [dim]Tip: reduce sample count or optimize /eval/run on server side.[/dim]")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", "") if isinstance(body, dict) else str(body)
        except Exception:
            detail = exc.response.text

        console.print(f"   [red]✗ Server returned HTTP {status} for /eval/run.[/red]")
        if detail:
            console.print(f"   [red]Detail:[/red] {detail}")
        console.print("   [dim]Check API logs to see the stack trace of the failing metric/model call.[/dim]")
        sys.exit(1)
    finally:
        stop_event.set()
        await heartbeat_task

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
