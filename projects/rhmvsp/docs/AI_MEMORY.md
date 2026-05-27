# RHMVSP Project AI Long-Term Memory and Guidelines

## 1. Problem Definition and Research Objectives (要解决的问题和研究目的)

Problem: The Reliable Hazardous Materials Multi-modal Vehicle Scheduling Problem (RHMVSP) addresses the complex logistics of transporting hazardous materials across a multi-modal network (road and rail) under supply uncertainty and time-varying risk factors.
Objectives: The research aims to formulate a mixed-integer programming model that minimizes total expected logistics costs (including vehicle deployment, transportation, hub transfer, and recourse penalties for unmet demand) while ensuring strict time and risk reliability bounds. The goal is to provide a robust routing and scheduling solution that dynamically adapts to uncertain supply at origins and time-dependent population exposure along routes.

## 2. Model Architecture and Main Components (模型架构与主要部件)

Multi-modal Network: Routes strictly follow a Road-Rail-Road transfer sequence. Pure road or pure rail routes are invalid.
Fuzzy Supply Uncertainty: Origin supply is modeled using TrapezoidalFuzzy variables instead of fixed-seed SAA scenarios. The expected supply dictates delivery capacity and recourse costs.
Time-Varying Risk and Consequence:

1. Accident probability relies on a time-varying traffic multiplier reflecting congestion and fatigue.
2. Accident consequence relies on a time-varying population exposure multiplier reflecting diurnal activity patterns.
   Continuous Reliability Metrics: Both time reliability and risk reliability are modeled as continuous uncertain measures and evaluated dynamically using a ReliabilityAnalyzer.

## 3. Algorithm Design Architecture and Main Components (算法设计架构与主要部件)

Dantzig-Wolfe Decomposition: The compact routing formulation is decomposed into a Master Problem (route selection) and a Pricing Problem (route generation).
Column Generation (CG): Iteratively solves the LP relaxation. The Master Problem provides dual variables, and the Pricing Problem finds new columns with negative reduced costs.
Pricing Subproblem (ESPPRC): Solved via a bidirectional Label Correcting algorithm. It uses ng-route relaxation for elementarity checks and strict state dominance rules (evaluating reduced cost, time, risk, and ng-memory).
Branch-and-Price (B&P): Wraps CG in a Branch-and-Bound tree to find exact integer solutions. It employs dynamic Ryan-Foster and hub activation branching.
Performance Optimizations: Uses aggressive label pruning (e.g., max 30-50 labels per node) and early break mechanisms in label joining to prevent combinatorial explosion in exact searches. Configurable early extraction of heuristic integer solutions at the root node is used for demo and small instances.

## 4. Literature Search and Update Rules (文献搜寻与更新法则)

Find Literature: Always prioritize searching for usable references in the following directories:
/Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/minecraft/2_paper_en_2026_spring/ref/
/Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/paper_old/
/Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/paper/
Update Literature: If new references are added, they must be instantly written to /Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/minecraft/2_paper_en_2026_spring/latex/cas-refs.bib and properly cited using \cite{...} in the main body of /Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/minecraft/2_paper_en_2026_spring/latex/main.tex.

## 5. Code and Experiment Verification Update Rules (代码与实验验证更新法则)

Planning and Artifact Workflow: Before modifying any code or document, ALWAYS generate or update an implementation plan and wait for user approval. Once the user approves, instantiate a task list to track real-time progress during execution. Finally, after completing the modifications, summarize all changes by updating the walkthrough document.
Execution Workflow: After every code modification, you do NOT need to run the experiments yourself. Instead, simply notify the user that the code is ready to be run. Once the user executes the experiment, analyze whether the results are reasonable and correct.
Document Updating: If the experiment results are reasonable, you must update the contents of /Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/minecraft/2_paper_en_2026_spring/latex/main.tex and /Users/frances/Library/CloudStorage/OneDrive-Personal/Papers/minecraft/2_paper_en_2026_spring/framework.md to reflect the latest codebase. Note that you must edit framework.md instead of framework.docx because .docx is a binary file format that AI cannot safely edit directly without risking corruption. The user will manually export the updated Markdown to Word.
Strict Formatting Constraints for Added/Modified Text:
Hyphens vs Dashes: Hyphens (-) are allowed (useful for constructing words like time-dependent, multi-modal, Branch-and-Price, mixed-integer, etc.). However, en-dashes (-- or –) and em-dashes (--- or —) are strictly forbidden.
Zero semicolons: Semicolons are entirely omitted or replaced with commas/periods.
Zero quotes: All quotation marks are completely avoided.
Forbidden Academic Cliches (Chinese): Words like 弥合, 闭环, 范式, 跃迁, 内生化, 劣化 are strictly avoided in framework.docx.
Forbidden Academic Cliches (English): Phrases like bridge the gap (use address this gap or similar), closed loop, paradigm, leap, internalize/internalization (use incorporate, embed, or model), and deteriorate/deterioration (use increase or degradation) are strictly avoided in main.tex.
Plain and Direct Language: Both Chinese and English texts must use plain, direct, and simple language. Avoid overly complex sentence structures or decorative rhetoric.
Active Verbs and Adverbs in LaTeX: In main.tex, make active use of strong verbs and descriptive adverbs to make the English writing dynamic and clear.
Varied Vocabulary (活用同义词): Dynamically alternate between different verbs and adverbs that convey similar meanings to avoid repetitive vocabulary (e.g., alternate between "use" and "deploy", or "totally" and "absolutely").
Language separation: Use Chinese only for .docx, and English only for .tex.

## 6. AI Workspace and Utility Files (AI工具文件管理)

Utility Scripts: Any non-project functional files generated by the AI (such as Python scripts for modifying .tex or .md files, refactoring scripts, etc.) MUST be placed in the `ai_utils/` directory at the project root. If this directory does not exist, you must create it. Do not clutter the project root or source directories with one-off AI utility scripts.
