<!-- SYNC IMPACT REPORT
Version change: N/A → 1.0.0
List of modified principles: N/A (new constitution)
Added sections: All principles and sections for Physical AI & Humanoid Robotics project
Removed sections: N/A
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs:
- RATIFICATION_DATE: Initial adoption date to be determined
-->
# AI/Spec-Driven Book — Physical AI & Humanoid Robotics Constitution

## Core Principles

### AI-Native Workflow
AI-native workflow using Spec-Kit Plus and Claude Code for all development tasks

### Specification-Driven Development
Specification-driven chapter generation with no writing without spec; every section must be generated or refined using lus

### Technical Accuracy
Technical accuracy grounded in robotics, AI, and physical automation with content accuracy validated through internal review

### Clarity and Accessibility
Clarity and accessibility for students, educators, and developers with consistent writing style across all chapters

### Consistency and Standards
Consistency of formatting, structure, and terminology across chapters with all chapters following Docusaurus Markdown structure

### Transparent Version Control
Transparent version control using Git and GitHub with clear commit messages and optional toolchain integration: MCP tools for structured automation

## Content Constraints

Only Markdown files allowed in the book body (.md or .mdx); No chapter may be written before the Specification is complete; No chapter may be added without a corresponding task in spec/tasks.md; Book must remain buildable (npm run build) at all times; Directory structure must remain clean and follow Docusaurus conventions; All images must be stored in static/img/<chapter-name>; No unverified or unrelated technical content allowed

## Deployment and Success Criteria

Deployment: GitHub Pages CI/CD; Key standards: Version control with clear commit messages, Docusaurus builds locally and deploys without errors; Success criteria: Full book deployed and accessible via GitHub Pages, Readers can navigate clearly from intro → advanced topics, Hackathon requirement met: AI/spec-driven workflow demonstrated end-to-end

## Governance

All chapters must be generated using specification-driven prompts in Claude Code; Constitution supersedes all other practices; Amendments require documentation and approval; All PRs/reviews must verify compliance with Docusaurus build requirements and specification-driven workflow; Book must remain buildable at all times

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): Initial adoption date to be determined | **Last Amended**: 2025-12-08