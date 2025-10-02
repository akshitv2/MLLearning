---
title: Agentic
nav_order: 18
parent: Notes
layout: default
---

# Agentic AI

## Table of Contents
- [Core Idea](#core-idea)
  - [AI Agent Definition](#ai-agent-definition)
  - [Characteristics of an AI Agent](#characteristics-of-an-ai-agent)
- [Architecture](#architecture)
  - [Base Model](#base-model)
  - [Memory System](#memory-system)
  - [Tool-Use/Actuator Layer](#tool-useactuator-layer)
  - [Planning Loop](#planning-loop)
  - [Self-Correction](#self-correction)
  - [Reasoning Techniques](#reasoning-techniques)
- [Multi-Agent Systems](#multi-agent-systems)
  - [Hierarchical Systems](#hierarchical-systems)
  - [Cooperative Systems](#cooperative-systems)
  - [Orchestration and Communication](#orchestration-and-communication)
  - [Emergent Behavior](#emergent-behavior)
- [Implementation](#implementation)
  - [Data and Knowledge Foundation](#data-and-knowledge-foundation)
  - [Infrastructure Optimization](#infrastructure-optimization)
  - [Model-Agnostic Architectures](#model-agnostic-architectures)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
  - [Key Metrics](#key-metrics)
  - [Robustness and Reliability](#robustness-and-reliability)
  - [Transparency and Explainability](#transparency-and-explainability)
- [Applications](#applications)
  - [Enterprise Use Cases](#enterprise-use-cases)
  - [Vertical Applications](#vertical-applications)

## Core Idea

### AI Agent Definition
An AI agent observes its environment, makes decisions using internal logic (often powered by Large Language Models, LLMs), and takes actions to achieve predefined goals. The LLM provides reasoning, while the agent adds structure and intent to execute plans.

### Characteristics of an AI Agent
- **Autonomy**: Operates independently without constant human oversight.
- **Perception**: Gathers and interprets environmental data.
- **Reasoning/Decision-Making**: Uses LLMs to evaluate options and select actions based on goals, context, and prior results.
- **Goal-Directedness & Planning**: Understands objectives, formulates, and executes plans.
- **Action/Execution**: Interacts with external systems (e.g., tools, APIs, file I/O) to impact the environment.
- **Learning and Adaptation**: Improves performance through experience and feedback.

## Architecture

### Base Model
The LLM or specialized model (e.g., vision transformer) serves as the agent's reasoning engine, processing inputs and generating decisions.

### Memory System
- **Short-Term Memory**: Stores task-relevant data (e.g., conversation history, recent tool outputs) within the LLM's context window. Limited by token count, causing older data to be forgotten.
- **Long-Term Memory**: Uses persistent storage (e.g., vector databases with Retrieval-Augmented Generation, RAG) for past experiences, facts, and preferences.

### Tool-Use/Actuator Layer
Connects the LLM’s reasoning to external actions via tools like:
- Web browsing/search for real-time data.
- Code interpreters for running/debugging code.
- APIs for interacting with services (e.g., email, CRM).
- File I/O for reading/writing documents.

### Planning Loop
The Sense-Think-Act Loop drives the agent’s iterative process:
1. **Sense**: Observes the environment (e.g., user input, tool outputs).
2. **Think**: LLM breaks down goals into manageable steps (e.g., for "write a 500-word stock market trends report," it plans research, data storage, and drafting).
3. **Act**: Executes actions via tools or outputs.

### Self-Correction
Handles failures by:
- Perceiving errors (e.g., failed tool calls).
- Reasoning about causes using the LLM.
- Refining the plan (e.g., adjusting queries, debugging code).

### Reasoning Techniques
| Technique | Acronym | Description | Goal |
|-----------|---------|-------------|------|
| Chain-of-Thought | CoT | Generates intermediate reasoning before final action. | Improves accuracy and transparency. |
| Tree-of-Thought | ToT | Explores multiple reasoning paths, backtracking from poor choices. | Enhances planning and decision quality. |
| Reasoning and Acting | ReAct | Interleaves reasoning, tool use, and observation in a structured loop. | Enables seamless goal-directed actions. |

## Multi-Agent Systems

### Hierarchical Systems
- **Structure**: A leader agent plans and delegates tasks to specialized sub-agents.
- **Workflow**: Leader assigns tasks, sub-agents execute and report back.
- **Advantage**: Efficient, controlled task management.

### Cooperative Systems
- **Structure**: Peer agents with complementary skills collaborate without a fixed hierarchy.
- **Workflow**: Agents share information, negotiate tasks, or compare results.
- **Advantage**: Robust, innovative solutions through diverse perspectives.

### Orchestration and Communication
Enables multi-agent coordination via:
- **Shared Memory/Whiteboard**: Centralized data structure for task updates.
- **Message Passing**: Direct agent communication using protocols (e.g., JSON messages).
- **Task Management Systems**: Tracks sub-tasks and dependencies.

### Emergent Behavior
Complex, unexpected behaviors arise from simple agent interactions, creating outcomes greater than the sum of individual actions.

## Implementation

### Data and Knowledge Foundation
- **Data Types**:
  - **Structured**: Databases, transactional records.
  - **Unstructured**: Documents, reports (accessed via vector databases for RAG).
- **Governance**: Ensures clean, secure, access-controlled data compliant with regulations (e.g., GDPR, HIPAA).

### Infrastructure Optimization
- **Microservices**: Small, API-callable services for granular actions.
- **Clean APIs**: Well-documented, low-latency APIs for efficient tool use.
- **Real-Time Interaction**: Low-latency infrastructure to support fast Sense-Think-Act cycles.

### Model-Agnostic Architectures
- **Interchangeable Components**: Treats LLMs and frameworks as plug-and-play to avoid vendor lock-in.
- **Standardized Interfaces**: Adapts agent calls to various models (e.g., Gemini, GPT, Llama) for flexibility.

## Evaluation and Benchmarking

### Key Metrics
| Metric | Description | Importance |
|--------|-------------|------------|
| Completion Rates | % of tasks completed without human intervention. | Measures autonomy and goal achievement. |
| Accuracy | Correctness of outputs for defined tasks. | Ensures reliable results. |
| Efficiency | Time and resource usage (e.g., tokens, API calls). | Critical for scalability and cost. |
| Intermediate Step Success | Success rate of sub-tasks/tool calls. | Identifies bottlenecks in planning. |

### Robustness and Reliability
- **Testing Scenarios**: Simulates unexpected variables, system failures, or adversarial prompts.
- **Red Teaming**: Tests resistance to malicious inputs to ensure safety.

### Transparency and Explainability
- **Decision Logging**: Records Sense-Think-Act steps (data, reasoning, actions, outcomes).
- **Purpose**:
  - **Debugging**: Identifies failure points.
  - **Compliance**: Meets regulatory requirements.
  - **Trust**: Builds confidence through auditable logic.

## Applications

### Enterprise Use Cases
- **Software Engineering**: Code generation, testing, debugging, project management.
- **Knowledge Work**: Research, data analysis, report generation, financial modeling.
- **Customer-Facing**: Customer service, personalized advising, sales outreach.

### Vertical Applications
- **Healthcare**: Diagnostics, patient care, administrative automation.
- **Finance**: Trading, fraud detection, compliance.
- **E-commerce/Retail**: Personalized assistants, supply chain optimization.
- **Manufacturing**: Process optimization, predictive maintenance.
- **Education**: Personalized tutoring, content generation.