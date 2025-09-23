---
title: LLM and Embedding Model Selection Guide
nav_order: 12
parent: Notes
layout: default
---

# LLM and Embedding Model Selection Guide

This guide provides a concise, tabular overview of key LLM and embedding models to simplify selection for your use case, with a focus on essential details: variants, sizes, features, access, pros/cons, and usage scenarios. Less critical data (e.g., niche models like BLOOM, GPT-NeoX, Vicuna, BERT) is excluded for clarity. Amazon Titan models are retained for AWS relevance. Pros/cons and usage recommendations are enhanced with practical insights based on performance, cost, and ecosystem fit, current as of September 2025.

## LLM Model Families

| Model Family | Key Variants | Parameter Sizes | Key Features | Access/Availability | Pros | Cons | When to Use |
|--------------|--------------|------------------|--------------|---------------------|------|------|-------------|
| GPT (OpenAI) | GPT-5, GPT-4o, GPT-4o mini, GPT-3.5 | Up to trillions | Multimodal (text, image, audio, video); advanced reasoning, coding, agentic workflows; powers ChatGPT. | OpenAI API; paid subscriptions (limited free tiers). | Top accuracy; robust API/tools; regular updates. | High cost; closed-source; peak-time latency. | General-purpose AI for reasoning, coding, multimodal apps. GPT-5/4o for complex workflows; mini/3.5 for cost-sensitive tasks. |
| Claude (Anthropic) | Claude 4 (Opus, Sonnet, Haiku), Claude 3.5 | Not disclosed (est. billions) | Constitutional AI for safety; multimodal (image/text); coding/enterprise; 1M token context (beta). | Anthropic API; AWS Bedrock; Google Vertex AI. | Strong safety; massive context; good for reasoning/coding. | Slower inference; limited free access; less multimodal depth. | Safety-critical enterprise apps (e.g., legal reviews). Opus for reasoning; Sonnet/Haiku for fast support bots. |
| Gemini (Google) | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash | Up to billions | Multimodal (text, image, audio, video); tool use; reasoning/coding; 2M token context. | Google AI Studio; Vertex AI; free tiers. | Large context; free/low-cost; Google ecosystem. | Weaker non-English; free tier limits; less agentic. | Multimodal projects (e.g., content moderation). Pro for analysis; Flash for mobile apps. |
| Gemma (Google) | Gemma 3 (1B-27B), Gemma 2 (9B-27B) | 1B-27B | Open-source; lightweight; text generation; multilingual; local inference. | Hugging Face; Google Vertex AI. | Efficient for edge; customizable; no API cost. | Lower performance; needs hardware; limited multimodal. | Lightweight local apps (e.g., mobile text gen). Smaller for constrained devices; larger for research. |
| LLaMA (Meta) | LLaMA 4 (Scout, Maverick), LLaMA 3.2 (1B-90B), LLaMA 3.1 (8B-405B) | 1B-405B | Open-source; multimodal (image/text); reasoning/coding/multilingual; 10M token context. | Hugging Face; Meta downloads. | Highly customizable; huge context; community support. | Compute-heavy; commercial licensing issues; uneven safety. | Open-source research; multimodal/long-context (e.g., RAG). Smaller for on-device; larger for high-performance. |
| Grok (xAI) | Grok 4 (Heavy), Grok 3, Grok 2, Grok 1.5 | Up to 314B | Multimodal (image/text); real-time search; coding/reasoning/agents; 128K token context. | xAI API; Grok platform (free limited; paid advanced). | Real-time data (e.g., X/Twitter); uncensored; dynamic agents. | Less mature ecosystem; variable availability; costly for heavy use. | Real-time search apps; uncensored agents. Heavy for reasoning; lighter for quick queries. |
| Mistral (Mistral AI) | Mistral Medium 3, Mixtral 8x22B | Up to 141B (Mixtral 39B active) | MoE; multilingual; coding/math; low-latency; open-source options. | Mistral API; Hugging Face. | Efficient scaling; cost-effective; strong math/coding. | Less multimodal; unstable API; smaller community. | Low-latency multilingual tasks; MoE for workloads. Mixtral for enterprise; open for fine-tuning. |
| Command (Cohere) | Command A, Command R+, Command R (7B) | 7B-104B | Enterprise-focused; multilingual; RAG with citations; tool use; 256K token context. | Cohere API; AWS Bedrock. | Transparent citations; scalable; multilingual. | Enterprise pricing; less creative; limited open-source. | Enterprise RAG/tool-use (e.g., CRM). R+ for docs; smaller for chatbots. |
| DeepSeek (DeepSeek) | DeepSeek V3.1, DeepSeek-R1 | Up to 671B | MoE; reasoning/math/coding; open-source (MIT); 128K token context. | DeepSeek API; Hugging Face. | Top open-source math; free license; efficient MoE. | Compute-heavy; smaller ecosystem; variable multilingual. | Open-source math/coding research. V3.1 for large inference. |
| Qwen (Alibaba) | Qwen3 (0.5B-235B), Qwen2.5 | 0.5B-235B | MoE; multilingual (12+ languages); coding/vision/audio; open-source. | Alibaba Cloud; Hugging Face. | Strong non-English; multimodal; cloud scalability. | Ecosystem lock-in; privacy concerns; weaker English. | Multilingual/multimodal global apps (e.g., e-commerce). |
| Ernie (Baidu) | Ernie 4.5, Ernie X1 | Billions (unspecified) | MoE; powers Ernie chatbot; open-sourced in 2025. | Baidu API. | Chinese dominance; integrated search; open-source push. | Limited global access; weaker non-Asian; proprietary core. | Chinese-market chatbots/search. |
| Falcon (TII) | Falcon 3, Falcon 2, Falcon Mamba 7B | 1B-180B | Open-source; multilingual/multimodal; efficient long sequences. | Hugging Face. | Efficient inference; multimodal; community-driven. | Lower benchmarks; smaller ecosystem. | Open-source long-sequence/multimodal apps. |
| Granite (IBM) | Granite 3.2, Granite Vision, Granite Code | Up to 34B | Open-source (Apache); enterprise RAG/code/risk detection. | watsonx.ai; Hugging Face. | Compliance-focused; IBM integration; specialized variants. | Mid-tier performance; enterprise pricing. | Enterprise RAG/code with risk tools. |

## Embedding Models

| Model | Developer | Dimensions | Key Performance/Pricing | Use Cases | Pros | Cons | When to Use |
|-------|----------|------------|-------------------------|-----------|------|------|-------------|
| text-embedding-3 | OpenAI | 3072/1536 (adjustable) | High English accuracy; $0.02-$0.13/M tokens. | Enterprise search; semantic similarity. | Reliable; Matryoshka efficiency; API ease. | Costly at scale; English bias. | High-accuracy English RAG/chatbots; large for precision, small for speed. |
| voyage-3 | Voyage AI | 2048/512 (adjustable) | Top retrieval; $0.02-$0.18/M tokens; free trial. | Long-context/multilingual/code search. | Cutting-edge relevance; 32K tokens; domain versatility. | Higher pricing; API-only. | Code/long-doc RAG; code-3 for programming. |
| Embed v3 | Cohere | 1024/384 | 100+ languages; $0.12/M tokens. | Multilingual RAG. | Broad language support; fast light versions. | Less domain depth; enterprise focus. | Global multilingual search. |
| text-embedding-004 | Google | 768 (adjustable) | Multilingual; free with limits. | Cost-effective search. | Low/no cost; quick latency. | Modest accuracy; free limits. | Prototypes/low-latency apps. |
| Jina Embeddings v3 | Jina AI | 1024 (adjustable to 32) | Long text (8K); free tier + paid. | Long docs/multilingual. | Task-optimized; adjustable; multilingual. | Newer, less tested. | Long-document embeddings. |
| Stella v5 | Dun Zhang | 1024 | Top open on MTEB; free (MIT). | Multilingual retrieval. | Benchmark leader; customizable. | Needs fine-tuning setup. | Open-source multilingual RAG. |
| Amazon Titan Embeddings | AWS | 1536 | Scalable; pay-per-use. | Cloud-based RAG. | Scalable; AWS security; enterprise-ready. | Pay-per-use cost; ecosystem lock-in. | Scalable AWS RAG with security needs. |