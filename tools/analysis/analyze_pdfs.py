#!/usr/bin/env python3
"""
PDF Content Analysis Script for Sheily AI Documentation
Analyzes all PDFs in docs/guides/ directory
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
    import PyPDF2
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Install with: pip install PyPDF2 pdfplumber langdetect")
    exit(1)


class PDFAnalyzer:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.filename = self.pdf_path.name
        self.size_mb = round(self.pdf_path.stat().st_size / (1024 * 1024), 2)

    def extract_metadata_pypdf2(self) -> Dict[str, Any]:
        """Extract metadata using PyPDF2"""
        try:
            with open(self.pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata

                return {
                    "title": getattr(info, "title", None),
                    "author": getattr(info, "author", None),
                    "subject": getattr(info, "subject", None),
                    "creator": getattr(info, "creator", None),
                    "producer": getattr(info, "producer", None),
                    "pages": len(pdf_reader.pages),
                    "encrypted": pdf_reader.is_encrypted,
                }
        except Exception as e:
            return {"error": f"PyPDF2 metadata extraction failed: {str(e)}"}

    def extract_text_pdfplumber(self) -> Dict[str, Any]:
        """Extract text content using pdfplumber"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                text_content = []
                total_chars = 0
                total_words = 0

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(
                            {
                                "page": page_num + 1,
                                "text": page_text,
                                "chars": len(page_text),
                                "words": len(page_text.split()),
                            }
                        )
                        total_chars += len(page_text)
                        total_words += len(page_text.split())

                full_text = " ".join([page["text"] for page in text_content])

                return {
                    "total_pages": len(pdf.pages),
                    "total_chars": total_chars,
                    "total_words": total_words,
                    "text_content": text_content,
                    "full_text": full_text,
                    "extraction_success": True,
                }
        except Exception as e:
            return {
                "error": f"pdfplumber text extraction failed: {str(e)}",
                "extraction_success": False,
            }

    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text"""
        try:
            if len(text.strip()) < 10:
                return "unknown"
            return detect(text)
        except LangDetectException:
            return "unknown"

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content for themes and keywords"""
        if not text:
            return {"error": "No text content to analyze"}

        # Convert to lowercase for analysis
        text_lower = text.lower()

        # Technical keywords related to AI/ML
        ai_keywords = [
            "machine learning",
            "deep learning",
            "neural network",
            "artificial intelligence",
            "natural language processing",
            "nlp",
            "computer vision",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "transfer learning",
            "fine-tuning",
            "transformer",
            "bert",
            "gpt",
            "llm",
            "large language model",
            "embedding",
            "token",
            "attention",
            "pre-training",
            "fine-tuning",
        ]

        # Agent-related keywords
        agent_keywords = [
            "agent",
            "multi-agent",
            "autonomous agent",
            "intelligent agent",
            "agent system",
            "agent architecture",
            "agent communication",
            "agent coordination",
            "agent framework",
            "mcp",
            "model context protocol",
            "agent tools",
            "agent interoperability",
        ]

        # Federated Learning keywords
        fl_keywords = [
            "federated learning",
            "federated",
            "federation",
            "distributed learning",
            "privacy-preserving",
            "differential privacy",
            "secure aggregation",
            "client",
            "server",
            "centralized",
            "decentralized",
        ]

        # Academic keywords
        academic_keywords = [
            "abstract",
            "introduction",
            "methodology",
            "experiment",
            "results",
            "conclusion",
            "references",
            "citation",
            "paper",
            "conference",
            "journal",
            "publication",
            "research",
            "study",
            "analysis",
        ]

        def count_keywords(keywords: List[str]) -> Dict[str, int]:
            counts = {}
            for keyword in keywords:
                count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text_lower))
                if count > 0:
                    counts[keyword] = count
            return counts

        # Count occurrences
        ai_counts = count_keywords(ai_keywords)
        agent_counts = count_keywords(agent_keywords)
        fl_counts = count_keywords(fl_keywords)
        academic_counts = count_keywords(academic_keywords)

        # Determine primary category
        category_scores = {
            "AI/ML Research": sum(ai_counts.values()),
            "Agent Systems": sum(agent_counts.values()),
            "Federated Learning": sum(fl_counts.values()),
            "Academic Paper": sum(academic_counts.values()),
        }

        primary_category = max(category_scores.items(), key=lambda x: x[1])
        primary_category = (
            primary_category[0] if primary_category[1] > 0 else "General Technical"
        )

        # Extract potential title (first meaningful line)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        potential_title = lines[0] if lines else "Unknown"

        # Check for code content
        code_indicators = [
            "import ",
            "def ",
            "class ",
            "function",
            "const ",
            "let ",
            "var ",
        ]
        code_lines = sum(
            1
            for line in lines
            if any(indicator in line.lower() for indicator in code_indicators)
        )

        return {
            "primary_category": primary_category,
            "category_scores": category_scores,
            "potential_title": potential_title[:200],  # Limit title length
            "ai_keywords_found": ai_counts,
            "agent_keywords_found": agent_counts,
            "fl_keywords_found": fl_counts,
            "academic_keywords_found": academic_counts,
            "code_content_detected": code_lines > 0,
            "code_lines_estimate": code_lines,
            "has_references": "references" in text_lower
            or "bibliography" in text_lower,
            "has_abstract": "abstract" in text_lower,
            "has_figures": len(re.findall(r"figure\s+\d+", text_lower)) > 0,
            "has_tables": len(re.findall(r"table\s+\d+", text_lower)) > 0,
        }

    def analyze_pdf(self) -> Dict[str, Any]:
        """Complete PDF analysis"""
        print(f"🔍 Analyzing: {self.filename}")

        # Extract metadata
        metadata = self.extract_metadata_pypdf2()

        # Extract text content
        text_data = self.extract_text_pdfplumber()

        # Analyze content if extraction was successful
        content_analysis = {}
        if text_data.get("extraction_success"):
            full_text = text_data.get("full_text", "")
            content_analysis = self.analyze_content(full_text)

            # Detect language
            detected_lang = self.detect_language(full_text)
            content_analysis["detected_language"] = detected_lang

        # Compile final analysis
        analysis = {
            "filename": self.filename,
            "filepath": str(self.pdf_path),
            "file_size_mb": self.size_mb,
            "metadata": metadata,
            "text_extraction": text_data,
            "content_analysis": content_analysis,
            "analysis_timestamp": str(Path(__file__).stat().st_mtime),
        }

        return analysis


def analyze_all_pdfs(directory: str = "docs/guides") -> Dict[str, Any]:
    """Analyze all PDFs in the specified directory"""
    pdf_dir = Path(directory)

    if not pdf_dir.exists():
        return {"error": f"Directory {directory} does not exist"}

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        return {"error": f"No PDF files found in {directory}"}

    print(f"📚 Found {len(pdf_files)} PDF files in {directory}")
    print("=" * 60)

    results = {}
    summary_stats = {
        "total_files": len(pdf_files),
        "total_size_mb": 0,
        "categories": Counter(),
        "languages": Counter(),
        "extraction_success": 0,
        "extraction_failed": 0,
    }

    for pdf_file in sorted(pdf_files):
        analyzer = PDFAnalyzer(pdf_file)
        analysis = analyzer.analyze_pdf()

        results[analyzer.filename] = analysis

        # Update summary stats
        summary_stats["total_size_mb"] += analyzer.size_mb

        if analysis["text_extraction"].get("extraction_success"):
            summary_stats["extraction_success"] += 1
            category = analysis["content_analysis"].get("primary_category", "Unknown")
            summary_stats["categories"][category] += 1

            language = analysis["content_analysis"].get("detected_language", "unknown")
            summary_stats["languages"][language] += 1
        else:
            summary_stats["extraction_failed"] += 1

        print(f"✅ Completed: {analyzer.filename}")
        print("-" * 40)

    # Compile final report
    final_report = {
        "summary": summary_stats,
        "files": results,
        "analysis_info": {
            "directory": str(pdf_dir),
            "total_files_analyzed": len(results),
            "analysis_date": str(Path(__file__).stat().st_mtime),
            "tools_used": ["PyPDF2", "pdfplumber", "langdetect"],
        },
    }

    return final_report


def save_analysis_report(
    report: Dict[str, Any], output_file: str = "pdf_analysis_report.json"
):
    """Save the analysis report to a JSON file"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"💾 Report saved to: {output_file}")
    print(f"📊 Total PDFs analyzed: {report['summary']['total_files']}")
    print(f"📈 Total size: {report['summary']['total_size_mb']:.2f} MB")
    print(f"✅ Successful extractions: {report['summary']['extraction_success']}")
    print(f"❌ Failed extractions: {report['summary']['extraction_failed']}")


def print_summary_report(report: Dict[str, Any]):
    """Print a human-readable summary of the analysis"""
    summary = report["summary"]

    print("\n" + "=" * 80)
    print("📊 PDF ANALYSIS SUMMARY REPORT")
    print("=" * 80)

    print(f"📁 Directory: {report['analysis_info']['directory']}")
    print(f"📚 Total PDFs: {summary['total_files']}")
    print(f"💾 Total Size: {summary['total_size_mb']:.2f} MB")
    print(f"✅ Extractions Exitosas: {summary['extraction_success']}")
    print(f"❌ Extracciones Fallidas: {summary['extraction_failed']}")

    print(f"\n🏷️  Categorías Principales:")
    for category, count in summary["categories"].most_common():
        print(f"   • {category}: {count} PDFs")

    print(f"\n🌍 Idiomas Detectados:")
    for language, count in summary["languages"].most_common():
        lang_name = {"en": "English", "es": "Spanish", "unknown": "Unknown"}.get(
            language, language
        )
        print(f"   • {lang_name}: {count} PDFs")

    print(
        f"\n🛠️  Herramientas Usadas: {', '.join(report['analysis_info']['tools_used'])}"
    )
    print(f"📅 Fecha de Análisis: {report['analysis_info']['analysis_date']}")


if __name__ == "__main__":
    print("🚀 Starting PDF Content Analysis for Sheily AI Documentation")
    print("=" * 60)

    # Analyze all PDFs
    report = analyze_all_pdfs("docs/guides")

    if "error" in report:
        print(f"❌ Error: {report['error']}")
        exit(1)

    # Save detailed report
    save_analysis_report(report, "pdf_content_analysis_report.json")

    # Print summary
    print_summary_report(report)

    print("\n" + "=" * 80)
    print("✅ PDF Analysis Complete!")
    print("📄 Check 'pdf_content_analysis_report.json' for detailed results")
    print("=" * 80)
