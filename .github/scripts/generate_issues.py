#!/usr/bin/env python3
"""
GitHub Issues Generator for CogPrime Development Roadmap

This script generates detailed, actionable issues for the CogPrime development phases.
Each issue follows the established format with comprehensive descriptions, tasks,
acceptance criteria, priorities, and effort estimates.
"""

import argparse
import json
import requests
import sys
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Issue:
    """Represents a development issue with all required fields."""
    title: str
    description: str
    tasks: List[str]
    acceptance_criteria: List[str]
    priority: str
    estimated_effort: str
    dependencies: List[str]
    labels: List[str]
    phase: str


class CogPrimeIssueGenerator:
    """Generator for CogPrime development issues across all phases."""
    
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.base_url = "https://api.github.com"
        
    def create_issue(self, issue: Issue, prefix: str = "", dry_run: bool = False) -> bool:
        """Create an issue on GitHub or simulate creation in dry run mode."""
        
        # Format the issue title with optional prefix
        title = f"{prefix}{issue.title}" if prefix else issue.title
        
        # Create the issue body with proper formatting
        body = self._format_issue_body(issue)
        
        if dry_run:
            print(f"\n{'='*60}")
            print(f"DRY RUN - Would create issue:")
            print(f"Title: {title}")
            print(f"Labels: {', '.join(issue.labels)}")
            print(f"Body:\n{body}")
            print(f"{'='*60}")
            return True
            
        # Create issue via GitHub API
        url = f"{self.base_url}/repos/{self.repo}/issues"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
        data = {
            "title": title,
            "body": body,
            "labels": issue.labels
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            issue_data = response.json()
            print(f"✅ Created issue #{issue_data['number']}: {title}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create issue '{title}': {e}")
            return False
    
    def _format_issue_body(self, issue: Issue) -> str:
        """Format issue body with proper markdown structure."""
        
        body_parts = [
            f"**Description:**",
            issue.description,
            "",
            "**Tasks:**"
        ]
        
        for task in issue.tasks:
            body_parts.append(f"- [ ] {task}")
        
        body_parts.extend([
            "",
            "**Acceptance Criteria:**"
        ])
        
        for criteria in issue.acceptance_criteria:
            body_parts.append(f"- {criteria}")
        
        body_parts.extend([
            "",
            f"**Priority:** {issue.priority}",
            f"**Estimated Effort:** {issue.estimated_effort}",
            f"**Phase:** {issue.phase}",
            ""
        ])
        
        if issue.dependencies:
            body_parts.extend([
                "**Dependencies:**"
            ])
            for dep in issue.dependencies:
                body_parts.append(f"- {dep}")
            body_parts.append("")
        
        body_parts.extend([
            "---",
            "",
            "*This issue was automatically generated from the CogPrime development roadmap.*"
        ])
        
        return "\n".join(body_parts)
    
    def get_phase2_issues(self) -> List[Issue]:
        """Generate issues for Phase 2: Advanced Features."""
        return [
            Issue(
                title="Enhanced Cognitive Capabilities Implementation",
                description="""
Implement advanced cognitive processing capabilities that extend beyond the basic architecture established in Phase 1. This includes sophisticated reasoning patterns, enhanced memory systems, and improved decision-making algorithms that leverage the full potential of the CogPrime cognitive architecture.

The enhanced capabilities should integrate seamlessly with existing modules while providing significant improvements in cognitive flexibility, adaptation, and learning efficiency.
                """.strip(),
                tasks=[
                    "Implement advanced pattern recognition algorithms in reasoning module",
                    "Add sophisticated memory consolidation mechanisms",
                    "Create adaptive attention allocation systems",
                    "Develop enhanced goal hierarchies and planning capabilities",
                    "Implement cross-modal integration for sensory processing",
                    "Add cognitive flexibility metrics and monitoring",
                    "Create dynamic resource allocation algorithms",
                    "Implement advanced error correction and recovery mechanisms"
                ],
                acceptance_criteria=[
                    "Advanced reasoning capabilities demonstrate 30% improvement over baseline",
                    "Memory consolidation reduces storage requirements by 20% while maintaining accuracy",
                    "Attention allocation adapts dynamically to changing environmental demands",
                    "Goal planning handles multi-level hierarchies with temporal constraints",
                    "Cross-modal integration improves perception accuracy by 25%",
                    "System maintains real-time performance with enhanced capabilities",
                    "Comprehensive test coverage validates all new cognitive functions"
                ],
                priority="High",
                estimated_effort="8-12 days",
                dependencies=["Completed Phase 1 core implementation", "Performance baseline metrics"],
                labels=["phase-2", "enhancement", "cognitive-architecture"],
                phase="Phase 2"
            ),
            Issue(
                title="Self-Reflection Mechanisms Development",
                description="""
Develop comprehensive self-reflection and introspection capabilities that allow the system to monitor, evaluate, and optimize its own cognitive processes. This includes meta-cognitive awareness, performance monitoring, and self-improvement mechanisms.

The self-reflection system should provide insights into cognitive state, decision quality, and learning effectiveness while enabling autonomous optimization of cognitive strategies.
                """.strip(),
                tasks=[
                    "Design meta-cognitive monitoring architecture",
                    "Implement cognitive state introspection functions",
                    "Create decision quality assessment mechanisms",
                    "Develop performance tracking and analytics",
                    "Add self-optimization feedback loops",
                    "Implement cognitive bias detection and correction",
                    "Create reflective learning mechanisms",
                    "Add confidence estimation for cognitive outputs"
                ],
                acceptance_criteria=[
                    "Meta-cognitive monitoring captures key cognitive processes accurately",
                    "Decision quality metrics correlate with actual performance outcomes",
                    "Self-optimization improves system performance over time",
                    "Bias detection identifies and mitigates common cognitive biases",
                    "Reflective learning accelerates adaptation to new domains",
                    "Confidence estimates are well-calibrated with actual accuracy",
                    "System maintains efficiency while adding self-reflection capabilities"
                ],
                priority="High",
                estimated_effort="10-14 days",
                dependencies=["Enhanced cognitive capabilities", "Monitoring framework"],
                labels=["phase-2", "meta-cognition", "self-improvement"],
                phase="Phase 2"
            ),
            Issue(
                title="Meta-Learning Systems Implementation",
                description="""
Implement sophisticated meta-learning capabilities that enable the system to learn how to learn more effectively. This includes transfer learning, few-shot learning, and adaptive learning strategies that improve with experience across different domains and tasks.

The meta-learning system should optimize learning algorithms, transfer knowledge between domains, and adapt learning strategies based on task characteristics and performance feedback.
                """.strip(),
                tasks=[
                    "Design meta-learning architecture and interfaces",
                    "Implement transfer learning mechanisms across domains",
                    "Create few-shot learning capabilities",
                    "Develop adaptive learning rate and strategy selection",
                    "Add knowledge distillation and compression techniques",
                    "Implement curriculum learning optimization",
                    "Create meta-parameter optimization systems",
                    "Add cross-domain knowledge transfer validation"
                ],
                acceptance_criteria=[
                    "Transfer learning reduces time to competency by 50% in new domains",
                    "Few-shot learning achieves 80% accuracy with minimal examples",
                    "Adaptive strategies outperform fixed learning approaches",
                    "Knowledge distillation maintains performance with reduced model size",
                    "Curriculum learning optimizes learning sequence automatically",
                    "Meta-parameters adapt to task characteristics effectively",
                    "Cross-domain transfer preserves relevant knowledge while avoiding negative transfer"
                ],
                priority="High",
                estimated_effort="12-16 days",
                dependencies=["Self-reflection mechanisms", "Learning module enhancements"],
                labels=["phase-2", "meta-learning", "transfer-learning"],
                phase="Phase 2"
            ),
            Issue(
                title="Cross-Domain Integration Framework",
                description="""
Develop a comprehensive framework for integrating cognitive processing across different domains and modalities. This includes unified representation systems, cross-domain reasoning, and seamless knowledge transfer between specialized cognitive subsystems.

The integration framework should enable the system to leverage knowledge and capabilities from one domain to enhance performance in others, creating synergistic effects across the cognitive architecture.
                """.strip(),
                tasks=[
                    "Design unified representation framework for cross-domain concepts",
                    "Implement cross-modal attention and integration mechanisms",
                    "Create domain adaptation and alignment algorithms",
                    "Develop abstract concept mapping between domains",
                    "Add cross-domain reasoning and inference capabilities",
                    "Implement knowledge graph integration across modalities",
                    "Create domain-specific to general knowledge translation",
                    "Add validation for cross-domain knowledge consistency"
                ],
                acceptance_criteria=[
                    "Unified representations enable seamless cross-domain reasoning",
                    "Cross-modal integration improves multi-sensory task performance",
                    "Domain adaptation maintains 90% accuracy across different contexts",
                    "Abstract concept mapping enables knowledge generalization",
                    "Cross-domain reasoning produces logically consistent results",
                    "Knowledge graphs integrate information from multiple domains effectively",
                    "Domain translation preserves semantic meaning and relationships"
                ],
                priority="Medium",
                estimated_effort="14-18 days",
                dependencies=["Meta-learning systems", "Enhanced cognitive capabilities"],
                labels=["phase-2", "integration", "cross-domain"],
                phase="Phase 2"
            ),
            Issue(
                title="Relevance Optimization System",
                description="""
Implement advanced relevance optimization mechanisms that dynamically prioritize cognitive resources based on contextual importance, goal relevance, and environmental demands. This system should incorporate Vervaeke's relevance realization framework principles.

The relevance optimization should improve efficiency, reduce cognitive load, and enhance the system's ability to focus on what matters most in any given context while maintaining flexibility and adaptability.
                """.strip(),
                tasks=[
                    "Design relevance assessment and scoring algorithms",
                    "Implement dynamic attention allocation based on relevance",
                    "Create contextual importance weighting mechanisms",
                    "Develop goal-relevance alignment systems",
                    "Add environmental salience detection and processing",
                    "Implement relevance-based memory retrieval optimization",
                    "Create adaptive filtering based on relevance thresholds",
                    "Add relevance feedback and learning mechanisms"
                ],
                acceptance_criteria=[
                    "Relevance scoring accurately predicts task importance and outcomes",
                    "Dynamic attention allocation improves task performance by 35%",
                    "Contextual weighting adapts appropriately to changing environments",
                    "Goal-relevance alignment maintains focus on important objectives",
                    "Salience detection identifies critical environmental changes",
                    "Memory retrieval prioritizes relevant information effectively",
                    "Adaptive filtering reduces cognitive load without losing important information"
                ],
                priority="Medium",
                estimated_effort="10-14 days",
                dependencies=["Cross-domain integration", "Attention mechanisms"],
                labels=["phase-2", "relevance", "optimization"],
                phase="Phase 2"
            )
        ]
    
    def get_phase3_issues(self) -> List[Issue]:
        """Generate issues for Phase 3: Framework Integration."""
        return [
            Issue(
                title="Vervaeke Framework Components Integration",
                description="""
Integrate John Vervaeke's 4E cognition framework components into the CogPrime architecture. This includes implementing embodied, embedded, enacted, and extended cognition principles that enhance the system's contextual awareness and meaning-making capabilities.

The integration should preserve the theoretical rigor of Vervaeke's framework while making it computationally tractable within the existing cognitive architecture.
                """.strip(),
                tasks=[
                    "Implement embodied cognition principles in sensorimotor processing",
                    "Add embedded cognition context awareness mechanisms",
                    "Create enacted cognition active perception systems",
                    "Develop extended cognition tool use and environmental coupling",
                    "Integrate salience landscape navigation algorithms",
                    "Implement perspectival and participatory knowing frameworks",
                    "Add transformative and conformative processing modes",
                    "Create cognitive-emotional integration following Vervaeke's model"
                ],
                acceptance_criteria=[
                    "Embodied cognition improves sensorimotor integration and spatial reasoning",
                    "Embedded cognition enables context-sensitive behavior adaptation",
                    "Enacted cognition supports active perception and exploration strategies",
                    "Extended cognition effectively utilizes environmental affordances",
                    "Salience landscape navigation guides attention and action selection",
                    "Perspectival knowing enables multiple viewpoint integration",
                    "Transformative processing facilitates insight and creative problem-solving",
                    "Cognitive-emotional integration enhances decision-making quality"
                ],
                priority="High",
                estimated_effort="16-20 days",
                dependencies=["Phase 2 completion", "Vervaeke framework documentation"],
                labels=["phase-3", "vervaeke-framework", "4e-cognition"],
                phase="Phase 3"
            ),
            Issue(
                title="Historical Context Integration System",
                description="""
Develop systems for integrating historical context and temporal understanding into cognitive processing. This includes episodic memory organization, temporal reasoning, and the ability to understand and utilize historical patterns in decision-making.

The historical context system should enable the architecture to learn from past experiences, understand temporal relationships, and make decisions informed by historical patterns and trends.
                """.strip(),
                tasks=[
                    "Design temporal knowledge representation frameworks",
                    "Implement episodic memory with temporal indexing",
                    "Create historical pattern recognition algorithms",
                    "Develop temporal reasoning and inference mechanisms",
                    "Add causal relationship detection across time",
                    "Implement historical context-aware decision making",
                    "Create temporal abstraction and generalization systems",
                    "Add historical knowledge validation and consistency checking"
                ],
                acceptance_criteria=[
                    "Temporal representations capture time-dependent relationships accurately",
                    "Episodic memory enables efficient retrieval of context-relevant experiences",
                    "Historical patterns inform and improve future predictions",
                    "Temporal reasoning handles complex time-dependent scenarios",
                    "Causal detection identifies meaningful cause-effect relationships",
                    "Historical context improves decision quality by 25%",
                    "Temporal abstractions enable pattern generalization across time scales"
                ],
                priority="High",
                estimated_effort="14-18 days",
                dependencies=["Vervaeke framework integration", "Memory system enhancements"],
                labels=["phase-3", "temporal-reasoning", "historical-context"],
                phase="Phase 3"
            ),
            Issue(
                title="Meaning-Making Systems Development",
                description="""
Implement comprehensive meaning-making capabilities that enable the system to construct, understand, and utilize meaning across different contexts and domains. This includes semantic understanding, contextual interpretation, and the generation of meaningful responses to novel situations.

The meaning-making system should integrate cognitive, emotional, and contextual factors to create rich, nuanced understanding that goes beyond simple pattern matching or rule-based processing.
                """.strip(),
                tasks=[
                    "Design semantic representation and processing frameworks",
                    "Implement contextual meaning construction algorithms",
                    "Create multi-level meaning integration systems",
                    "Develop symbolic-subsymbolic meaning bridges",
                    "Add emotional-cognitive meaning synthesis",
                    "Implement cultural and social context understanding",
                    "Create meaning validation and consistency mechanisms",
                    "Add adaptive meaning refinement based on feedback"
                ],
                acceptance_criteria=[
                    "Semantic representations capture nuanced meaning relationships",
                    "Contextual meaning adapts appropriately to different situations",
                    "Multi-level integration creates coherent meaning hierarchies",
                    "Symbolic-subsymbolic bridges enable rich meaning processing",
                    "Emotional synthesis enhances meaning depth and relevance",
                    "Cultural context understanding improves communication and interaction",
                    "Meaning validation ensures consistency and coherence",
                    "Adaptive refinement improves meaning accuracy over time"
                ],
                priority="High",
                estimated_effort="18-22 days",
                dependencies=["Historical context integration", "Semantic processing foundations"],
                labels=["phase-3", "meaning-making", "semantics"],
                phase="Phase 3"
            ),
            Issue(
                title="Relevance Realization Implementation",
                description="""
Implement Vervaeke's relevance realization framework as a core cognitive mechanism that guides attention, memory, and decision-making. This system should dynamically determine what is relevant in any given context and adaptively allocate cognitive resources accordingly.

The relevance realization system should operate across multiple time scales and cognitive levels, from immediate sensory processing to long-term goal planning and knowledge integration.
                """.strip(),
                tasks=[
                    "Implement Vervaeke's relevance realization algorithms",
                    "Create multi-scale relevance assessment mechanisms",
                    "Develop adaptive relevance threshold systems",
                    "Add relevance-guided attention and memory mechanisms",
                    "Implement relevance-based knowledge integration",
                    "Create relevance learning and adaptation systems",
                    "Add relevance propagation across cognitive modules",
                    "Implement relevance-action coupling mechanisms"
                ],
                acceptance_criteria=[
                    "Relevance realization accurately identifies contextually important information",
                    "Multi-scale assessment operates effectively from milliseconds to hours",
                    "Adaptive thresholds optimize relevance sensitivity for different contexts",
                    "Attention and memory systems effectively utilize relevance guidance",
                    "Knowledge integration prioritizes relevant information appropriately",
                    "Learning systems improve relevance assessment accuracy over time",
                    "Relevance propagation maintains consistency across cognitive modules",
                    "Action coupling enables relevance-informed behavior selection"
                ],
                priority="High",
                estimated_effort="16-20 days",
                dependencies=["Meaning-making systems", "Attention mechanisms"],
                labels=["phase-3", "relevance-realization", "vervaeke-framework"],
                phase="Phase 3"
            ),
            Issue(
                title="Meta-Cognitive Synthesis Framework",
                description="""
Develop a comprehensive meta-cognitive synthesis framework that integrates insights from cognitive science, philosophy of mind, and artificial intelligence to create a unified approach to meta-cognition within the CogPrime architecture.

This framework should enable higher-order thinking about thinking, self-awareness, and the ability to reason about cognitive processes themselves, creating a foundation for advanced artificial general intelligence capabilities.
                """.strip(),
                tasks=[
                    "Design meta-cognitive architecture and interfaces",
                    "Implement higher-order thinking about thinking capabilities",
                    "Create self-awareness and introspection mechanisms",
                    "Develop cognitive process reasoning and analysis",
                    "Add meta-cognitive strategy selection and optimization",
                    "Implement recursive meta-cognitive processing",
                    "Create meta-cognitive knowledge representation systems",
                    "Add meta-cognitive learning and adaptation mechanisms"
                ],
                acceptance_criteria=[
                    "Meta-cognitive architecture enables reasoning about cognitive processes",
                    "Higher-order thinking demonstrates insight and understanding beyond first-order cognition",
                    "Self-awareness mechanisms provide accurate introspective capabilities",
                    "Cognitive process analysis improves system understanding and optimization",
                    "Strategy selection adapts meta-cognitive approaches to different tasks",
                    "Recursive processing enables deep meta-cognitive analysis",
                    "Knowledge representation captures meta-cognitive insights effectively",
                    "Learning mechanisms improve meta-cognitive capabilities over time"
                ],
                priority="Medium",
                estimated_effort="20-24 days",
                dependencies=["Relevance realization", "All Phase 3 components"],
                labels=["phase-3", "meta-cognition", "synthesis"],
                phase="Phase 3"
            )
        ]
    
    def get_phase4_issues(self) -> List[Issue]:
        """Generate issues for Phase 4: Optimization."""
        return [
            Issue(
                title="Performance Tuning and Optimization",
                description="""
Implement comprehensive performance optimization across all cognitive modules to ensure real-time operation and efficient resource utilization. This includes algorithmic optimizations, memory management improvements, and computational efficiency enhancements.

The optimization should maintain or improve cognitive capabilities while significantly reducing computational requirements and improving response times.
                """.strip(),
                tasks=[
                    "Profile all cognitive modules for performance bottlenecks",
                    "Implement algorithmic optimizations for critical paths",
                    "Add memory pooling and efficient allocation strategies",
                    "Optimize tensor operations and neural network computations",
                    "Implement parallel processing for independent cognitive operations",
                    "Add caching mechanisms for frequently accessed patterns",
                    "Optimize attention mechanisms for reduced computational complexity",
                    "Implement just-in-time compilation for performance-critical code"
                ],
                acceptance_criteria=[
                    "Overall system performance improves by 50% compared to Phase 3",
                    "Memory usage reduces by 30% while maintaining functionality",
                    "Real-time operation achieved for all interactive components",
                    "Critical cognitive cycles complete within 10ms",
                    "Parallel processing utilizes available cores effectively",
                    "Caching reduces repeated computation overhead by 40%",
                    "Optimized algorithms maintain accuracy while improving speed",
                    "JIT compilation provides measurable performance benefits"
                ],
                priority="High",
                estimated_effort="12-16 days",
                dependencies=["Complete Phase 3 implementation", "Performance profiling tools"],
                labels=["phase-4", "optimization", "performance"],
                phase="Phase 4"
            ),
            Issue(
                title="Resource Management System",
                description="""
Develop sophisticated resource management capabilities that dynamically allocate computational resources, memory, and attention based on current cognitive demands and system constraints. This includes priority-based scheduling, resource monitoring, and adaptive allocation strategies.

The resource management system should ensure optimal utilization of available resources while preventing resource conflicts and maintaining system stability under varying computational loads.
                """.strip(),
                tasks=[
                    "Design dynamic resource allocation architecture",
                    "Implement priority-based computational scheduling",
                    "Create memory management with garbage collection optimization",
                    "Add attention resource allocation algorithms",
                    "Implement adaptive load balancing across cognitive modules",
                    "Create resource monitoring and alerting systems",
                    "Add resource constraint handling and graceful degradation",
                    "Implement resource usage prediction and preallocation"
                ],
                acceptance_criteria=[
                    "Dynamic allocation responds to changing computational demands within 1ms",
                    "Priority scheduling ensures critical cognitive operations complete on time",
                    "Memory management prevents leaks and optimizes usage patterns",
                    "Attention allocation maximizes cognitive effectiveness given resource constraints",
                    "Load balancing maintains system responsiveness under high load",
                    "Monitoring provides real-time visibility into resource utilization",
                    "Graceful degradation maintains core functionality under resource pressure",
                    "Prediction systems enable proactive resource management"
                ],
                priority="High",
                estimated_effort="14-18 days",
                dependencies=["Performance optimization", "System monitoring"],
                labels=["phase-4", "resource-management", "scheduling"],
                phase="Phase 4"
            ),
            Issue(
                title="Scalability Improvements",
                description="""
Implement scalability enhancements that enable the cognitive architecture to handle increased complexity, larger knowledge bases, and more sophisticated cognitive tasks without proportional increases in computational requirements.

The scalability improvements should include distributed processing capabilities, hierarchical organization, and efficient scaling algorithms that maintain cognitive coherence as system complexity grows.
                """.strip(),
                tasks=[
                    "Design distributed cognitive processing architecture",
                    "Implement hierarchical knowledge organization systems",
                    "Add modular cognitive component scaling mechanisms",
                    "Create efficient data structures for large-scale cognitive operations",
                    "Implement distributed attention and memory systems",
                    "Add cognitive load balancing across processing nodes",
                    "Create scalable learning algorithms that maintain efficiency",
                    "Implement fault tolerance and recovery mechanisms for distributed systems"
                ],
                acceptance_criteria=[
                    "Distributed processing scales linearly with additional computational nodes",
                    "Hierarchical organization handles knowledge bases 10x larger efficiently",
                    "Modular scaling allows selective enhancement of cognitive capabilities",
                    "Data structures maintain O(log n) performance for cognitive operations",
                    "Distributed systems maintain cognitive coherence across nodes",
                    "Load balancing prevents computational hotspots and resource conflicts",
                    "Learning algorithms scale to handle significantly larger datasets",
                    "Fault tolerance ensures system reliability in distributed configurations"
                ],
                priority="Medium",
                estimated_effort="16-20 days",
                dependencies=["Resource management", "Distributed computing framework"],
                labels=["phase-4", "scalability", "distributed-systems"],
                phase="Phase 4"
            ),
            Issue(
                title="System Refinement and Integration",
                description="""
Conduct comprehensive system refinement to ensure all cognitive components work together seamlessly and efficiently. This includes integration testing, performance validation, and refinement of interfaces between different cognitive modules.

The refinement process should identify and resolve any remaining integration issues, optimize inter-module communication, and ensure the overall system operates as a coherent cognitive architecture.
                """.strip(),
                tasks=[
                    "Conduct comprehensive integration testing across all modules",
                    "Refine inter-module communication protocols and interfaces",
                    "Optimize data flow and processing pipelines",
                    "Add comprehensive system validation and verification procedures",
                    "Implement end-to-end cognitive scenario testing",
                    "Refine error handling and recovery mechanisms",
                    "Optimize configuration and parameter tuning systems",
                    "Add comprehensive documentation and API refinement"
                ],
                acceptance_criteria=[
                    "Integration testing validates seamless operation across all modules",
                    "Inter-module communication operates efficiently without bottlenecks",
                    "Data pipelines maintain consistency and accuracy throughout processing",
                    "Validation procedures verify correct operation under diverse conditions",
                    "End-to-end testing demonstrates coherent cognitive behavior",
                    "Error handling prevents system failures and enables graceful recovery",
                    "Configuration systems enable easy optimization and customization",
                    "Documentation provides complete guidance for system usage and extension"
                ],
                priority="High",
                estimated_effort="10-14 days",
                dependencies=["Scalability improvements", "All Phase 4 optimizations"],
                labels=["phase-4", "integration", "refinement"],
                phase="Phase 4"
            ),
            Issue(
                title="Meta-Strategic Enhancement System",
                description="""
Develop advanced meta-strategic capabilities that enable the system to optimize its cognitive strategies, learning approaches, and problem-solving methods based on performance feedback and environmental demands.

The meta-strategic system should continuously improve the system's ability to select appropriate cognitive strategies, adapt to new challenges, and optimize its overall cognitive performance across diverse tasks and domains.
                """.strip(),
                tasks=[
                    "Design meta-strategic reasoning and planning architecture",
                    "Implement strategy effectiveness evaluation mechanisms",
                    "Create adaptive strategy selection algorithms",
                    "Add meta-learning for strategy optimization",
                    "Implement strategy transfer and generalization systems",
                    "Create strategic knowledge representation and storage",
                    "Add real-time strategy adaptation based on performance feedback",
                    "Implement meta-strategic memory and experience integration"
                ],
                acceptance_criteria=[
                    "Meta-strategic reasoning improves cognitive strategy selection by 40%",
                    "Effectiveness evaluation accurately predicts strategy performance",
                    "Adaptive selection chooses optimal strategies for given contexts",
                    "Meta-learning improves strategy effectiveness over time",
                    "Strategy transfer applies successful approaches to new domains",
                    "Knowledge representation captures strategic insights effectively",
                    "Real-time adaptation responds to changing task demands",
                    "Memory integration enables strategic learning from experience"
                ],
                priority="Medium",
                estimated_effort="18-22 days",
                dependencies=["System refinement", "Meta-cognitive frameworks from Phase 3"],
                labels=["phase-4", "meta-strategy", "optimization"],
                phase="Phase 4"
            )
        ]
    
    def generate_all_issues(self, phase: str, prefix: str = "", dry_run: bool = False) -> bool:
        """Generate issues for the specified phase(s)."""
        
        all_issues = []
        
        if phase == "all" or phase == "phase2":
            all_issues.extend(self.get_phase2_issues())
            
        if phase == "all" or phase == "phase3":
            all_issues.extend(self.get_phase3_issues())
            
        if phase == "all" or phase == "phase4":
            all_issues.extend(self.get_phase4_issues())
        
        if not all_issues:
            print(f"No issues found for phase: {phase}")
            return False
        
        success_count = 0
        total_count = len(all_issues)
        
        print(f"Generating {total_count} issues for phase(s): {phase}")
        
        for issue in all_issues:
            if self.create_issue(issue, prefix, dry_run):
                success_count += 1
            else:
                print(f"Failed to create issue: {issue.title}")
        
        print(f"\nCompleted: {success_count}/{total_count} issues created successfully")
        return success_count == total_count


def main():
    """Main entry point for the issue generator."""
    
    parser = argparse.ArgumentParser(description="Generate CogPrime development issues")
    parser.add_argument("--phase", required=True, 
                       choices=["all", "phase2", "phase3", "phase4"],
                       help="Phase to generate issues for")
    parser.add_argument("--prefix", default="", 
                       help="Prefix for issue titles")
    parser.add_argument("--dry-run", type=str, default="false",
                       help="Dry run mode - don't actually create issues")
    parser.add_argument("--token", required=True,
                       help="GitHub API token")
    parser.add_argument("--repo", required=True,
                       help="GitHub repository (owner/repo)")
    
    args = parser.parse_args()
    
    # Convert dry_run string to boolean
    dry_run = args.dry_run.lower() == "true"
    
    # Create generator and run
    generator = CogPrimeIssueGenerator(args.repo, args.token)
    success = generator.generate_all_issues(args.phase, args.prefix, dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()