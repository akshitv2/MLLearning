---
title: Agentic
nav_order: 18
parent: Notes
layout: default
---

# Agentic

# Core Idea

## AI agent:

- Observes environment, makes decisions based on these observations, and its own internal logic
- take actions to achieve a predefined goal.
- Agentic AI often uses Generative AI (specifically LLMs) as its Reasoning/Decision-Making engine. The LLM provides
  the "intelligence," but the Agent provides the "will" and the "structure" to execute a plan.

## Characteristics of an AI Agent

These characteristics define what an AI needs to be considered an "agent" in the modern sense:

- ## Autonomy
    - This is the agent's ability to operate without constant human supervision or intervention.

- ## Perception
    - Perception is the agent's mechanism for gathering and interpreting data from the environment.

- ## Reasoning/Decision-Making
    - This is the agent's internal process of weighing options and selecting an optimal action given its goal and its
      current perception.
        - Done by means of large language model
        - The LLM uses its training and context to decide:
            - What is the next logical step?
            - Which tool should I use?
            - How does the last action's result affect my overall plan?

- ## Goal-Directedness & Planning
    - An agent is fundamentally defined by its objective. It must be able to:
    - Understand the goal
    - formulate a plan
    - execute the plan
- ## Action/Execution
    - This is the process of interacting with external systems, tools, or the environment. An agent's intelligence is
      useless if it can't affect the world. Actions typically involve:
        - Calling external Tools (e.g., a web browser, a code interpreter, a database query).
        - Outputting information (e.g., writing a report, sending an email).
        - Making changes to a digital system (e.g., editing a file, deploying code).
- ## Learning and Adaptation
    - Improving performance based on experience and feedback

# Architecture

## Base Model (The Brain)

The Large Language Model (LLM) or other foundation model (like a specialized vision transformer), agent's Reasoning
Engine

## Memory System

- Implementation: The agent's prompt and current context window within the LLM.
- Role: Stores immediate information relevant to the current task—the conversation history, the results of the last few
  tool calls, and the agent's current plan.
- Limitation: It is limited by the maximum token count of the LLM. Once the context window is full, the agent starts
  to "forget" the oldest information, making it poor for long-term tasks.
- ### Long-Term Memory
    - Persistent storage, often a Vector Database used in conjunction with Retrieval-Augmented Generation (RAG).
    - Role: Stores past experiences, learned facts, preferences, and key output

## Tool-Use/Actuator Layer

- Definition: The Actuator Layer is the agent's mechanism for interacting with the external world. It’s the bridge
  connecting the LLM's internal thought process to executable functions.
- Allows the agent to take actions—its "hands." Tools can include:
    - Web Browsing/Search: Getting up-to-date information.
    - Code Interpreters: Running code, analyzing data, debugging.
    - APIs: Interacting with services (e.g., sending email, accessing a CRM, fetching weather data).
    - File I/O: Reading and writing documents.

## The Planning Loop (The Agent's "Thought Process")

The agent is defined by its continuous, iterative process—often called the Sense-Think-Act Loop—which repeats until the
goal is achieved.

When given a complex objective (e.g., "Research the top 5 stock market trends and write a 500-word report"), the agent's
first step in the loop is to use the LLM to break the goal down into a sequence of smaller, manageable steps (the plan).

- Initial Thought: "I need to research trends, summarize them, and write a report."
- Plan Step 1: Use the Google Search Tool for "top 5 stock market trends 2025."
- Plan Step 2: Use the File I/O Tool to store the search results.
- Plan Step 3: Use the LLM to summarize and draft the report.

## Self-Correction and Iterative Refinement

- A key differentiator for an agent is its ability to handle failure. If a tool call fails (e.g., a search query returns
  no relevant results, or code execution throws an error), the agent does not quit.
- Perception: The error message is perceived.
- Reasoning: The LLM analyzes the error.
- Refinement: The agent adjusts the plan, perhaps trying a different tool, modifying the search query, or attempting to
  debug the failed code. This constant iterative refinement is crucial for robustness.

| Technique	           | Acronym	 | Description	                                                                                                                                                                      | Primary Goal                                                                   |
|----------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Chain-of-Thought     | CoT      | The agent is prompted to first generate an intermediate reasoning step or "thought" before arriving  at the final answer or action.                                               | Improve Accuracy and show the agent's logic.                                   |
| Tree-of-Thought      | ToT      | Generalizes CoT by exploring multiple possible reasoning paths simultaneously, effectively allowingthe agent to branch out its thinking and backtrack from bad decisions.         | Improve Quality and enable better planning via lookahead                       |
| Reasoning and Acting | ReAct    | The agent interleaves a Thought (internal reasoning) with an Action (tool use), and an Observation (tool result). This is a structured way to implement the entire planning loop. | Enable Goal-Directed Action by integrating tool use seamlessly with reasoning. |

## Multi-Agent Systems and Collaboration

Multi-Agent Systems involve two or more agents working together to solve problems that would be difficult or impossible
for a single agent to handle alone.

There are two primary ways to structure a team of AI agents:

- ## Hierarchical Systems (Vertical)
    - This architecture is like a corporate structure or a military chain of command.
    - **Structure**: A "Leader" or "Orchestrator" Agent sits at the top. This agent is responsible for the high-level
      planning, goal decomposition, and progress tracking.
    - **Delegation**: The leader delegates specific, specialized tasks to Sub-Agents (also called Worker Agents).
    - **Workflow**: The leader hands down a task, the sub-agent executes it, and the sub-agent reports the result or
      completion status back to the leader.
    - **Advantage**: Efficiency and Control. Dependencies are clearly managed by the leader, minimizing redundant work
      and keeping the entire system focused.
- ## Cooperative/Collaborative Systems (Horizontal)

This architecture is like a brainstorming session or a peer-review team, often referred to as "Agent Societies."

- Structure: Agents possess different, often complementary, expertise but operate as peers without a fixed hierarchy.
- Collaboration: Agents interact directly with each other, sharing information, asking for help, and refining each
  other's work towards a shared goal.
- Workflow: Agents negotiate or communicate using standardized protocols (see Orchestration below) to decide who does
  what, or they might even perform the same task and compare results for robustness.
- Advantage: Robustness and Innovation. The lack of a single leader makes the system less prone to catastrophic failure,
  and diverse perspectives can lead to more creative solutions.

- ## Orchestration and Communication
- For multiple agents to work together effectively, they need standardized ways to talk, share work, and manage
  dependencies. This is handled by Orchestration and Communication protocols.
- Goal: To enable agents to share information, manage dependencies, and coordinate actions.
    - Shared Memory/Whiteboard: A centralized, accessible data structure where agents can post their outputs, update the
      status of sub-tasks, and read information needed for their next step
    - Message Passing: Agents communicate directly with each other using defined protocols (like an API call or a
      structured JSON message).
    - Task Management Systems: Tools (internal or external) used to maintain a global view of the project's status,
      tracking which agent is working on which sub-task and noting dependencies between tasks.
- ## Emergent Behavior
- Definition: Emergent Behavior is the complex, often unexpected, system behavior that arises from the interaction of
  multiple relatively simple agents, even if that behavior was not explicitly programmed into any single agent.
- Nature: The whole is greater (and sometimes different) than the sum of its parts. Individual agents follow simple
  rules, but their collective interaction creates intricate, global patterns.

# Implementation

## Data and Knowledge Foundation

- ### Designing Clean, Governed Data Ecosystems:
    - Agents need access to two main types of data:
        - Structured Data: Databases, tables, and transactional records (e.g., customer purchase history).
        - Unstructured Data: Documents, manuals, internal reports, and emails (often accessed via Vector Databases for
          RAG).
    - The Governance Component: Crucially, the data must be clean (accurate, consistent) and governed (secure,
      access-controlled). An agent, being autonomous, must only be able to access data it is authorized for, adhering to
      privacy rules (like GDPR or HIPAA)
- ### Infrastructure Optimization
    - Modernizing Core Systems (APIs, Microservices): An agent’s "Action/Execution" relies on robust interfaces to
      external
      systems. Traditional monolithic applications are too rigid. Agents thrive when interacting with:
        - Microservices: Small, independent services that are easily callable via APIs, allowing the agent to perform
          granular actions (e.g., one service for "check inventory," another for "place order").
        - Clean, Documented APIs: The agent's LLM reasons about which tool to call by reading its description.
          Well-defined, low-latency APIs are essential for efficient and successful tool use.
    - Ensuring Seamless and Real-time Interaction: The latency of the entire process (Perceive → Reason → Act) is
      critical. Infrastructure needs to handle:
        - The high computational cost of the LLM/reasoning step.
        - Fast, reliable execution of multiple, consecutive API calls by the agent.
- ### Model Agnostic Architectures
    - Relying on a single LLM provider (e.g., one company's specific model) creates business risk. Smart agent design
      prevents this.
    - Designing Systems for Interchangeable Components: A model agnostic architecture treats the LLM (the "Brain") and
      the specific agent framework as plug-and-play components.
    - Avoiding Vendor Lock-in: The goal is to separate the agent's core logic (the planning loop, memory system, and
      tool definitions) from the specific model that powers its reasoning. This is typically achieved using:
        - Standardized Interfaces: Designing a unified interface or adapter layer that can translate the agent's call
          into the specific API format required by Google's Gemini, OpenAI's GPT, or an open-source Llama model.
        - Flexibility: This allows an organization to easily swap out the foundation model based on cost, performance,
          security, or the task requirement (e.g., using a smaller, cheaper model for simple tasks and a
          state-of-the-art model for complex planning).

- # Evaluation and Benchmarking

| Metric                           | Description                                                                                                                    | Why It Matters for Agents                                                                                                                   |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Completion Rates                 | The percentage of times an agent successfully finishes a task (i.e., reaches the final goal state) without human intervention. | The ultimate measure of Autonomy and Goal-Directedness. If the rate is low, the agent fails to justify its existence.                       |
| Accuracy                         | For tasks with a definitive correct output (e.g., coding, data extraction), this measures the correctness of the final result. | Ensures the agent isn't just completing the task but doing it correctly.                                                                    |
| Efficiency (Time/Resource Usage) | Time to Completion (latency) and the Cost (e.g., tokens consumed, API calls made, compute resources used).                     | High efficiency is key for scalability and cost-effectiveness. An agent that takes too long or costs too much is not viable for production. |
| Intermediate Step Success        | Measuring the success rate of individual sub-tasks or tool calls within the planning loop.                                     | Helps pinpoint bottlenecks or specific tools/reasoning steps that frequently fail.                                                          |

## Robustness and Reliablity

- handle the unpredictability of the real world. This goes beyond simple accuracy.
- Testing under Unexpected Scenarios: This involves introducing variables the agent wasn't explicitly trained or
  designed for.
- System Failures: Simulating real-world IT issues.
- Adversarial Prompts: Testing the agent's resistance to malicious or confusing inputs designed to make it break its
  rules or perform harmful actions (a form of Red Teaming).

## Transparency and Explainability (XAI)

Autonomy is powerful, but it must be balanced with accountability. Explainable AI (XAI) is essential for building trust
and enabling debugging.

- Documenting the Agent's Decision-Making Process: Every step of the Sense-Think-Act Loop should be logged and
  auditable. This "paper trail" should include:
    - Perception: What data was gathered (e.g., search results, API response).
    - Reasoning/Thought: The LLM's explicit Chain-of-Thought or ReAct internal monologue (why it chose the next action).
    - Action: Which tool was called, and with what parameters.
    - Outcome: The result or error from the action.
- Purpose of Audit and Oversight:
    - Debugging: When an agent fails, the logs are the only way for a human developer to understand where the planning
      went wrong.
    - Compliance: In regulated industries (like finance or healthcare), agents must prove why they made a specific
      decision, often for regulatory review.
    - Trust: Allowing human supervisors to understand and verify the agent's logic is vital for adoption.

## Enterprise and Business Transformation

Agentic AI's value comes from its ability to automate complex, multistep processes that previously required human
planning and execution. This is fundamentally changing how work is done across various sectors.

- Software Engineering Agents: Automated code generation, testing, debugging, and project management.
- Knowledge Work Automation: Agents for research, data analysis, report generation, and automated financial modeling.
- Customer-Facing Agents: Advanced customer service, personalized financial advising, and automated sales outreach.
- Vertical Applications (Deep Dive):
    - Healthcare: Diagnostics, personalized patient care, administrative task automation.
    - Finance: Automated trading, fraud detection, and regulatory compliance.
    - E-commerce/Retail: Personalized shopping assistants, supply chain optimization.
