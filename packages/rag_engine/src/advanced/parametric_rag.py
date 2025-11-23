#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric Retrieval Augmented Generation (Parametric RAG)
Based on "Parametric Retrieval Augmented Generation" (2025)

Implements the new RAG paradigm that injects knowledge directly into LLM parameters
instead of appending to context. Features:
- Document Augmentation (rewriting + QA generation)
- Parametric Document Encoding (LoRA per document)
- Retrieve-Update-Generate workflow
- 29-36% faster inference vs in-context RAG
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ParametricDocument:
    """Represents a document with its parametric representation"""

    doc_id: str
    original_text: str
    augmented_data: Dict[str, Any]  # Rewrites + QA pairs
    lora_path: str  # Path to saved LoRA parameters
    metadata: Dict[str, Any]


@dataclass
class ParametricRAGResult:
    """Result of Parametric RAG inference"""

    query: str
    retrieved_docs: List[str]
    merged_lora_path: str
    generated_answer: str
    inference_time: float
    metadata: Dict[str, Any]


class AugmentedDocumentDataset(Dataset):
    """Dataset for parametric document training"""

    def __init__(
        self, augmented_data: List[Dict[str, str]], tokenizer, max_length: int = 1024
    ):
        self.data = augmented_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format: [document + question + answer]
        text = f"Document: {item['document']}\nQuestion: {item['question']}\nAnswer: {item['answer']}"

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].flatten(),
            "attention_mask": encodings["attention_mask"].flatten(),
            "labels": encodings["input_ids"].flatten(),
        }


class ParametricRAG:
    """
    Parametric Retrieval Augmented Generation

    Implements the Retrieve-Update-Generate paradigm:
    1. Document Augmentation (offline)
    2. Parametric Document Encoding (offline)
    3. Online inference with parameter merging
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        retriever=None,  # Should be injected
        device: str = "auto",
    ):
        """
        Initialize Parametric RAG system

        Args:
            base_model: Base LLM for parametric encoding
            retriever: Retrieval system (BM25, dense, hybrid)
            device: Device to use
        """
        self.base_model_name = base_model
        self.retriever = retriever
        self.device = (
            device if device != "auto" else ("cuda" if self._has_cuda() else "cpu")
        )

        # Initialize components
        self.tokenizer = None
        self.base_model = None
        self.lora_config = None

        # Document registry
        self.parametric_documents: Dict[str, ParametricDocument] = {}

        self._initialize_components()

    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _initialize_components(self):
        """Initialize model and tokenizer"""
        if not PEFT_AVAILABLE or not TORCH_AVAILABLE:
            print("Warning: PEFT or PyTorch not available, Parametric RAG disabled")
            return

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            # Prepare for LoRA
            self.base_model = prepare_model_for_kbit_training(self.base_model)

            # LoRA config for parametric encoding (paper settings)
            self.lora_config = LoraConfig(
                r=2,  # Low rank for efficiency (paper uses r=2)
                lora_alpha=32,  # Scaling factor
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            print(f"Initialized Parametric RAG with {self.base_model_name}")

        except Exception as e:
            print(f"Failed to initialize Parametric RAG: {e}")

    def augment_document(
        self, document: str, num_rewrites: int = 1, num_qa_pairs: int = 3
    ) -> Dict[str, Any]:
        """
        Document Augmentation phase: Generate rewrites and QA pairs

        Args:
            document: Original document text
            num_rewrites: Number of rewrites to generate
            num_qa_pairs: Number of QA pairs per document

        Returns:
            Augmented document data
        """
        augmented_data = {"original": document, "rewrites": [], "qa_pairs": []}

        # Generate rewrites
        for i in range(num_rewrites):
            # In practice, this would use the LLM to generate rewrites
            # For now, we'll use simple variations
            rewrite_prompt = (
                f"Rewrite the following text in a different way: {document}"
            )
            # TODO: Use LLM to generate actual rewrites
            rewrite = f"{document} (Rewritten version {i+1})"
            augmented_data["rewrites"].append(rewrite)

        # Generate QA pairs
        for i in range(num_qa_pairs):
            # In practice, this would use the LLM to generate QA pairs
            # For now, we'll use template-based generation
            question = f"What is described in document section {i+1}?"
            answer = f"The document describes: {document[:200]}..."
            augmented_data["qa_pairs"].append({"question": question, "answer": answer})

        return augmented_data

    def encode_parametric_document(
        self,
        doc_id: str,
        augmented_data: Dict[str, Any],
        output_dir: str = "parametric_docs",
    ) -> ParametricDocument:
        """
        Parametric Document Encoding: Train LoRA for this document

        Args:
            doc_id: Unique document identifier
            augmented_data: Augmented document data
            output_dir: Directory to save LoRA parameters

        Returns:
            ParametricDocument with trained LoRA
        """
        Path(output_dir).mkdir(exist_ok=True)
        lora_path = f"{output_dir}/{doc_id}_lora"

        # Prepare training data from augmented document
        training_data = []

        # Add rewrites + QA pairs combinations
        for rewrite in augmented_data["rewrites"]:
            for qa_pair in augmented_data["qa_pairs"]:
                training_data.append(
                    {
                        "document": rewrite,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                    }
                )

        # Create dataset
        train_dataset = AugmentedDocumentDataset(training_data, self.tokenizer)

        # Apply LoRA to base model
        lora_model = get_peft_model(self.base_model, self.lora_config)

        # Training arguments (following paper)
        training_args = TrainingArguments(
            output_dir=lora_path,
            num_train_epochs=1,  # Paper uses 1 epoch per document
            per_device_train_batch_size=1,  # Small batch for per-document training
            learning_rate=3e-4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=5,
            save_steps=100,
            fp16=self.device == "cuda",
            dataloader_num_workers=0,
            report_to=[],
            evaluation_strategy="no",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        print(f"Training parametric representation for document {doc_id}...")
        trainer.train()

        # Save LoRA parameters
        trainer.save_model(lora_path)
        self.tokenizer.save_pretrained(lora_path)

        # Create ParametricDocument
        parametric_doc = ParametricDocument(
            doc_id=doc_id,
            original_text=augmented_data["original"],
            augmented_data=augmented_data,
            lora_path=lora_path,
            metadata={
                "num_rewrites": len(augmented_data["rewrites"]),
                "num_qa_pairs": len(augmented_data["qa_pairs"]),
                "training_samples": len(training_data),
                "created_at": time.time(),
            },
        )

        # Register document
        self.parametric_documents[doc_id] = parametric_doc

        return parametric_doc

    def add_document(self, doc_id: str, document: str) -> ParametricDocument:
        """
        Add a document to the parametric knowledge base

        Args:
            doc_id: Unique document identifier
            document: Document text

        Returns:
            ParametricDocument ready for retrieval
        """
        # Augment document
        augmented_data = self.augment_document(document)

        # Encode parametrically
        parametric_doc = self.encode_parametric_document(doc_id, augmented_data)

        return parametric_doc

    def merge_lora_parameters(
        self, doc_ids: List[str], output_path: str = "merged_lora"
    ) -> str:
        """
        Merge LoRA parameters from multiple retrieved documents

        Args:
            doc_ids: List of document IDs to merge
            output_path: Path to save merged LoRA

        Returns:
            Path to merged LoRA parameters
        """
        if not doc_ids:
            return None

        try:
            from peft import PeftModel

            # Load base model
            merged_model = self.base_model

            # Merge LoRA parameters (following paper's merging strategy)
            for doc_id in doc_ids:
                if doc_id not in self.parametric_documents:
                    continue

                doc = self.parametric_documents[doc_id]
                lora_model = PeftModel.from_pretrained(merged_model, doc.lora_path)

                # Merge with scaling (α parameter from paper)
                scaling = 32  # From paper configuration
                merged_model = lora_model.merge_and_unload()

            # Save merged model
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

            return output_path

        except Exception as e:
            print(f"Failed to merge LoRA parameters: {e}")
            return None

    def generate_with_parametric_rag(
        self, query: str, top_k: int = 3, max_new_tokens: int = 256
    ) -> ParametricRAGResult:
        """
        Execute Parametric RAG: Retrieve → Update → Generate

        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate

        Returns:
            ParametricRAGResult with generated answer
        """
        start_time = time.time()

        # 1. Retrieve phase
        retrieved_docs = []
        if self.retriever:
            # Use injected retriever to get top-k documents
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        else:
            # Fallback: use all available documents
            retrieved_docs = list(self.parametric_documents.keys())[:top_k]

        # 2. Update phase: Merge parametric representations
        merged_lora_path = None
        if retrieved_docs:
            merged_lora_path = self.merge_lora_parameters(retrieved_docs)

        # 3. Generate phase: Use updated model
        if merged_lora_path:
            try:
                from peft import PeftModel

                # Load merged model
                parametric_model = PeftModel.from_pretrained(
                    self.base_model, merged_lora_path
                )

                # Tokenize query
                inputs = self.tokenizer(query, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = parametric_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                answer = generated_text[len(query) :].strip()  # Remove query prefix

            except Exception as e:
                print(f"Generation failed: {e}")
                answer = "Error: Failed to generate answer with parametric model"
        else:
            answer = "Error: No parametric documents available"

        inference_time = time.time() - start_time

        return ParametricRAGResult(
            query=query,
            retrieved_docs=retrieved_docs,
            merged_lora_path=merged_lora_path,
            generated_answer=answer,
            inference_time=inference_time,
            metadata={
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "num_retrieved": len(retrieved_docs),
            },
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for Parametric RAG
        Based on paper experimental results
        """
        return {
            "inference_speedup": {
                "vs_standard_rag": "29-36% faster",
                "vs_flare": "6x faster",
                "vs_dragin": "8x faster",
            },
            "storage_efficiency": {
                "per_document_mb": 4.72,  # 16-bit LoRA
                "scaling_factor": "Sub-linear with document count",
            },
            "benchmark_performance": {
                "2wqa_improvement": "+13.5% F1",
                "hotpotqa_improvement": "+8.2% F1",
                "popqa_improvement": "+15.1% F1",
                "cwq_improvement": "+12.3% F1",
            },
            "efficiency_break_even": {
                "queries_needed": "2x document count",
                "description": "Breaks even when queries > 2x documents",
            },
        }

    def save_parametric_corpus(self, filepath: str):
        """Save parametric document registry"""
        data = {
            doc_id: {
                "doc_id": doc.doc_id,
                "original_text": doc.original_text,
                "lora_path": doc.lora_path,
                "metadata": doc.metadata,
            }
            for doc_id, doc in self.parametric_documents.items()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_parametric_corpus(self, filepath: str):
        """Load parametric document registry"""
        with open(filepath, "r") as f:
            data = json.load(f)

        for doc_id, doc_data in data.items():
            # Recreate ParametricDocument (without augmented_data for space)
            self.parametric_documents[doc_id] = ParametricDocument(
                doc_id=doc_data["doc_id"],
                original_text=doc_data["original_text"],
                augmented_data={},  # Not stored for space
                lora_path=doc_data["lora_path"],
                metadata=doc_data["metadata"],
            )
