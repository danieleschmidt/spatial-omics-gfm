#!/usr/bin/env python3
"""Terragon Autonomous Value Discovery Engine.

Continuously discovers, scores, and prioritizes improvement opportunities
across code quality, security, performance, and feature development.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class ValueDiscoveryEngine:
    """Autonomous engine for discovering and scoring improvement opportunities."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        if self.config_path.exists() and yaml:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {
            "scoring": {
                "weights": {
                    "nascent": {"wsjf": 0.4, "ice": 0.3, "technicalDebt": 0.2, "security": 0.1}
                }
            },
            "project": {"maturity": "nascent"}
        }
        
    def _load_metrics(self) -> Dict:
        """Load value metrics and history."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {"executionHistory": [], "backlogMetrics": {}}
        
    def _save_metrics(self) -> None:
        """Save updated metrics."""
        if "repository" not in self.metrics:
            self.metrics["repository"] = {}
        self.metrics["repository"]["lastAssessment"] = datetime.now(timezone.utc).isoformat()
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    def discover_opportunities(self) -> List[Dict]:
        """Discover improvement opportunities from multiple sources."""
        opportunities = []
        
        # Git history analysis
        opportunities.extend(self._analyze_git_history())
        
        # Documentation analysis
        opportunities.extend(self._documentation_analysis())
        
        # Testing analysis  
        opportunities.extend(self._testing_analysis())
        
        # Structure analysis
        opportunities.extend(self._structure_analysis())
        
        return opportunities
        
    def _analyze_git_history(self) -> List[Dict]:
        """Analyze git history for TODO, FIXME, and technical debt indicators."""
        opportunities = []
        
        try:
            result = subprocess.run([
                "git", "grep", "-n", "-i", r"TODO\|FIXME\|HACK\|DEPRECATED"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts
                        opportunities.append({
                            "id": f"git-debt-{abs(hash(line))}",
                            "title": f"Address technical debt in {file_path}:{line_num}",
                            "description": content.strip(),
                            "category": "technical_debt",
                            "source": "git_history",
                            "files": [file_path],
                            "effort_hours": 1,
                            "impact_score": 5,
                            "confidence": 8,
                            "ease": 7
                        })
                    
        except subprocess.CalledProcessError:
            logger.debug("No TODO/FIXME comments found in git history")
            
        return opportunities
        
    def _documentation_analysis(self) -> List[Dict]:
        """Analyze documentation coverage and quality."""
        opportunities = []
        
        # Check for missing documentation files
        expected_docs = [
            ("CHANGELOG.md", "Create CHANGELOG for release tracking"),
            ("API.md", "Create API documentation"),
            ("docs/installation.md", "Create detailed installation guide"),
            ("docs/tutorials/", "Create getting started tutorials"),
            ("examples/", "Create usage examples")
        ]
        
        for doc_file, description in expected_docs:
            doc_path = self.repo_path / doc_file
            if not doc_path.exists():
                opportunities.append({
                    "id": f"docs-missing-{doc_file.lower().replace('/', '-').replace('.', '-')}",
                    "title": f"Add {doc_file}",
                    "description": description,
                    "category": "documentation",
                    "source": "documentation_analysis",
                    "files": [doc_file],
                    "effort_hours": 3,
                    "impact_score": 7,
                    "confidence": 9,
                    "ease": 8
                })
                
        return opportunities
        
    def _testing_analysis(self) -> List[Dict]:
        """Analyze test coverage and quality."""
        opportunities = []
        
        # Check if basic testing infrastructure exists
        tests_dir = self.repo_path / "tests"
        if not tests_dir.exists():
            opportunities.append({
                "id": "testing-infrastructure-setup",
                "title": "Set up testing infrastructure",
                "description": "Create tests/ directory with pytest configuration and basic test structure",
                "category": "testing",
                "source": "testing_analysis",
                "files": ["tests/"],
                "effort_hours": 4,
                "impact_score": 9,
                "confidence": 9,
                "ease": 7
            })
        
        # Check for src directory without tests
        src_dir = self.repo_path / "src"
        if not src_dir.exists():
            opportunities.append({
                "id": "python-package-structure",
                "title": "Create Python package structure",
                "description": "Set up src/ directory with proper Python package structure",
                "category": "foundation",
                "source": "structure_analysis",
                "files": ["src/"],
                "effort_hours": 6,
                "impact_score": 9,
                "confidence": 9,
                "ease": 6
            })
            
        return opportunities
        
    def _structure_analysis(self) -> List[Dict]:
        """Analyze project structure and identify missing components."""
        opportunities = []
        
        # Check for CI/CD workflows
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists() or len(list(workflows_dir.glob("*.yml"))) == 0:
            opportunities.append({
                "id": "cicd-workflows-setup",
                "title": "Set up CI/CD workflows",
                "description": "Create GitHub Actions workflows for testing, linting, and deployment",
                "category": "cicd",
                "source": "structure_analysis", 
                "files": [".github/workflows/"],
                "effort_hours": 5,
                "impact_score": 8,
                "confidence": 8,
                "ease": 7
            })
            
        # Check for Makefile or task runner
        if not (self.repo_path / "Makefile").exists():
            opportunities.append({
                "id": "development-automation",
                "title": "Add development automation (Makefile)",
                "description": "Create Makefile with common development tasks (test, lint, format, build)",
                "category": "developer_experience",
                "source": "structure_analysis",
                "files": ["Makefile"],
                "effort_hours": 2,
                "impact_score": 6,
                "confidence": 9,
                "ease": 9
            })
            
        return opportunities
        
    def score_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Score opportunities using WSJF + ICE + Technical Debt model."""
        weights = self.config.get("scoring", {}).get("weights", {})
        maturity = self.config.get("project", {}).get("maturity", "nascent")
        maturity_weights = weights.get(maturity, {
            "wsjf": 0.4, "ice": 0.3, "technicalDebt": 0.2, "security": 0.1
        })
        
        for opp in opportunities:
            # WSJF Score
            user_value = opp.get("impact_score", 5)
            time_criticality = 8 if opp.get("security_critical") else 5
            risk_reduction = 7 if opp.get("category") == "security" else 4
            opportunity_enablement = 8 if opp.get("category") == "foundation" else 3
            
            cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
            job_size = opp.get("effort_hours", 1)
            wsjf_score = cost_of_delay / max(job_size, 0.1)
            
            # ICE Score
            impact = opp.get("impact_score", 5)
            confidence = opp.get("confidence", 7)
            ease = opp.get("ease", 5)
            ice_score = impact * confidence * ease
            
            # Technical Debt Score
            debt_impact = 20 if opp.get("category") == "technical_debt" else 10
            debt_interest = 15 if opp.get("category") == "technical_debt" else 5
            hotspot_multiplier = 2.0 if opp.get("category") == "foundation" else 1.0
            tech_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
            
            # Composite Score
            composite_score = (
                maturity_weights.get("wsjf", 0.4) * min(wsjf_score / 10, 10) +
                maturity_weights.get("ice", 0.3) * min(ice_score / 100, 10) +
                maturity_weights.get("technicalDebt", 0.2) * min(tech_debt_score / 10, 10) +
                maturity_weights.get("security", 0.1) * (10 if opp.get("security_critical") else 5)
            )
            
            # Apply category boosts
            if opp.get("category") == "foundation":
                composite_score *= 1.8
            elif opp.get("category") == "testing":
                composite_score *= 1.5
                
            opp["scores"] = {
                "wsjf": round(wsjf_score, 2),
                "ice": ice_score,
                "technicalDebt": round(tech_debt_score, 2),
                "composite": round(composite_score, 2)
            }
            
        return sorted(opportunities, key=lambda x: x["scores"]["composite"], reverse=True)
        
    def update_backlog(self, opportunities: List[Dict]) -> None:
        """Update BACKLOG.md with discovered opportunities."""
        backlog_path = self.repo_path / "BACKLOG.md"
        
        if not backlog_path.exists():
            return
            
        content = backlog_path.read_text()
        
        # Find the section to update
        if "## 🚀 Next Best Value Items" in content:
            before_section = content.split("## 🚀 Next Best Value Items")[0]
            after_parts = content.split("---")
            after_section = "---" + after_parts[-1] if len(after_parts) > 1 else ""
        else:
            before_section = content
            after_section = ""
            
        # Generate new section
        new_section = self._generate_backlog_section(opportunities)
        
        # Update content
        updated_content = f"{before_section}## 🚀 Next Best Value Items\n\n{new_section}\n\n{after_section}"
        
        # Write back
        backlog_path.write_text(updated_content)
        
    def _generate_backlog_section(self, opportunities: List[Dict]) -> str:
        """Generate the backlog section content."""
        if not opportunities:
            return "*No high-value opportunities currently discovered. System will continue monitoring.*\n"
            
        # Top item
        top_item = opportunities[0]
        section = f"""### 🎯 Next Best Value Item
**[{top_item['id'].upper()}] {top_item['title']}**
- **Composite Score**: {top_item['scores']['composite']}
- **WSJF**: {top_item['scores']['wsjf']} | **ICE**: {top_item['scores']['ice']} | **Tech Debt**: {top_item['scores']['technicalDebt']}
- **Estimated Effort**: {top_item['effort_hours']} hours
- **Category**: {top_item['category'].replace('_', ' ').title()}
- **Description**: {top_item['description']}

"""
        
        # Top 10 items table
        section += "### 📋 Top 10 Backlog Items\n\n"
        section += "| Rank | ID | Title | Score | Category | Est. Hours |\n"
        section += "|------|-----|--------|---------|----------|------------|\n"
        
        for i, opp in enumerate(opportunities[:10], 1):
            category = opp['category'].replace('_', ' ').title()
            title = opp['title'][:50] + "..." if len(opp['title']) > 50 else opp['title']
            section += f"| {i} | {opp['id'][:8]} | {title} | {opp['scores']['composite']} | {category} | {opp['effort_hours']} |\n"
            
        # Discovery stats
        section += f"\n### 🔍 Discovery Summary\n\n"
        section += f"- **Total Opportunities Found**: {len(opportunities)}\n"
        
        categories = {}
        for opp in opportunities:
            cat = opp['category']
            categories[cat] = categories.get(cat, 0) + 1
            
        section += f"- **By Category**:\n"
        for cat, count in sorted(categories.items()):
            section += f"  - {cat.replace('_', ' ').title()}: {count}\n"
            
        if opportunities:
            section += f"- **Average Score**: {sum(opp['scores']['composite'] for opp in opportunities) / len(opportunities):.1f}\n"
            section += f"- **Total Estimated Effort**: {sum(opp['effort_hours'] for opp in opportunities)} hours\n"
        
        return section
        
    def run_discovery_cycle(self) -> Dict:
        """Run a complete discovery and scoring cycle."""
        logger.info("Starting autonomous value discovery cycle")
        
        # Discover opportunities
        opportunities = self.discover_opportunities()
        logger.info(f"Discovered {len(opportunities)} opportunities")
        
        # Score opportunities
        scored_opportunities = self.score_opportunities(opportunities)
        logger.info(f"Scored and ranked {len(scored_opportunities)} opportunities")
        
        # Update backlog
        self.update_backlog(scored_opportunities)
        logger.info("Updated BACKLOG.md with discovered opportunities")
        
        # Update metrics
        if "backlogMetrics" not in self.metrics:
            self.metrics["backlogMetrics"] = {}
            
        self.metrics["backlogMetrics"].update({
            "totalItems": len(scored_opportunities),
            "averageScore": sum(opp["scores"]["composite"] for opp in scored_opportunities) / max(len(scored_opportunities), 1),
            "lastDiscovery": datetime.now(timezone.utc).isoformat()
        })
        self._save_metrics()
        
        return {
            "opportunities_found": len(opportunities),
            "highest_score": scored_opportunities[0]["scores"]["composite"] if scored_opportunities else 0,
            "categories": list(set(opp["category"] for opp in opportunities)),
            "next_best_item": scored_opportunities[0] if scored_opportunities else None
        }


def main():
    """Main entry point for the discovery engine."""
    logging.basicConfig(level=logging.INFO)
    
    repo_path = Path.cwd()
    engine = ValueDiscoveryEngine(repo_path)
    
    try:
        results = engine.run_discovery_cycle()
        print(f"✅ Discovery complete: {results['opportunities_found']} opportunities found")
        if results['next_best_item']:
            print(f"🎯 Next best item: {results['next_best_item']['title']} (Score: {results['next_best_item']['scores']['composite']})")
        else:
            print("🎯 No opportunities discovered at this time")
    except Exception as e:
        logger.error(f"Discovery cycle failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()