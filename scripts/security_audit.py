#!/usr/bin/env python3
"""
Security audit script for DocsReview RAG application.
Runs comprehensive security scans using bandit, safety, and semgrep.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class SecurityAuditor:
    """Security auditor for the DocsReview RAG application."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize security auditor.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.results = {}
        self.critical_issues = []
        self.high_issues = []
        self.medium_issues = []
        self.low_issues = []
        
        # Security scan tools
        self.tools = {
            "bandit": self._check_bandit,
            "safety": self._check_safety,
            "semgrep": self._check_semgrep
        }
    
    def run_security_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive security audit.
        
        Returns:
            Audit results
        """
        logger.info("Starting security audit...")
        
        # Check if tools are available
        self._check_tool_availability()
        
        # Run security scans
        for tool_name, tool_func in self.tools.items():
            try:
                logger.info(f"Running {tool_name} scan...")
                result = tool_func()
                self.results[tool_name] = result
                logger.info(f"{tool_name} scan completed")
            except Exception as e:
                logger.error(f"{tool_name} scan failed: {e}")
                self.results[tool_name] = {"error": str(e)}
        
        # Analyze results
        self._analyze_results()
        
        # Generate report
        report = self._generate_report()
        
        logger.info("Security audit completed")
        return report
    
    def _check_tool_availability(self) -> None:
        """Check if security tools are available."""
        for tool in ["bandit", "safety", "semgrep"]:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                logger.info(f"{tool} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(f"{tool} is not available")
    
    def _check_bandit(self) -> Dict[str, Any]:
        """Run bandit security scan."""
        try:
            # Run bandit scan
            cmd = [
                "bandit",
                "-r", str(self.project_root / "src"),
                "-f", "json",
                "-ll"  # Low confidence, low severity
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Bandit found issues: {result.stderr}")
            
            # Parse results
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                return {
                    "status": "completed",
                    "results": bandit_results,
                    "issues_found": len(bandit_results.get("results", [])),
                    "high_severity": len([r for r in bandit_results.get("results", []) if r.get("issue_severity") == "HIGH"]),
                    "medium_severity": len([r for r in bandit_results.get("results", []) if r.get("issue_severity") == "MEDIUM"]),
                    "low_severity": len([r for r in bandit_results.get("results", []) if r.get("issue_severity") == "LOW"])
                }
            else:
                return {"status": "no_issues", "results": []}
                
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_safety(self) -> Dict[str, Any]:
        """Run safety dependency scan."""
        try:
            # Run safety scan
            cmd = ["safety", "check", "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Safety found vulnerabilities: {result.stderr}")
            
            # Parse results
            if result.stdout:
                safety_results = json.loads(result.stdout)
                return {
                    "status": "completed",
                    "results": safety_results,
                    "vulnerabilities_found": len(safety_results),
                    "critical": len([v for v in safety_results if v.get("severity") == "critical"]),
                    "high": len([v for v in safety_results if v.get("severity") == "high"]),
                    "medium": len([v for v in safety_results if v.get("severity") == "medium"]),
                    "low": len([v for v in safety_results if v.get("severity") == "low"])
                }
            else:
                return {"status": "no_vulnerabilities", "results": []}
                
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_semgrep(self) -> Dict[str, Any]:
        """Run semgrep SAST scan."""
        try:
            # Run semgrep scan
            cmd = [
                "semgrep",
                "--config=auto",
                str(self.project_root / "src"),
                "--json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Semgrep found issues: {result.stderr}")
            
            # Parse results
            if result.stdout:
                semgrep_results = json.loads(result.stdout)
                return {
                    "status": "completed",
                    "results": semgrep_results,
                    "issues_found": len(semgrep_results.get("results", [])),
                    "high_severity": len([r for r in semgrep_results.get("results", []) if r.get("extra", {}).get("severity") == "ERROR"]),
                    "medium_severity": len([r for r in semgrep_results.get("results", []) if r.get("extra", {}).get("severity") == "WARNING"]),
                    "low_severity": len([r for r in semgrep_results.get("results", []) if r.get("extra", {}).get("severity") == "INFO"])
                }
            else:
                return {"status": "no_issues", "results": []}
                
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_results(self) -> None:
        """Analyze security scan results."""
        for tool_name, result in self.results.items():
            if result.get("status") == "completed":
                if tool_name == "bandit":
                    self._analyze_bandit_results(result)
                elif tool_name == "safety":
                    self._analyze_safety_results(result)
                elif tool_name == "semgrep":
                    self._analyze_semgrep_results(result)
    
    def _analyze_bandit_results(self, result: Dict[str, Any]) -> None:
        """Analyze bandit results."""
        issues = result.get("results", {}).get("results", [])
        
        for issue in issues:
            severity = issue.get("issue_severity", "LOW")
            confidence = issue.get("issue_confidence", "LOW")
            
            if severity == "HIGH" and confidence == "HIGH":
                self.critical_issues.append(issue)
            elif severity == "HIGH":
                self.high_issues.append(issue)
            elif severity == "MEDIUM":
                self.medium_issues.append(issue)
            else:
                self.low_issues.append(issue)
    
    def _analyze_safety_results(self, result: Dict[str, Any]) -> None:
        """Analyze safety results."""
        vulnerabilities = result.get("results", [])
        
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low")
            
            if severity == "critical":
                self.critical_issues.append(vuln)
            elif severity == "high":
                self.high_issues.append(vuln)
            elif severity == "medium":
                self.medium_issues.append(vuln)
            else:
                self.low_issues.append(vuln)
    
    def _analyze_semgrep_results(self, result: Dict[str, Any]) -> None:
        """Analyze semgrep results."""
        issues = result.get("results", [])
        
        for issue in issues:
            severity = issue.get("extra", {}).get("severity", "INFO")
            
            if severity == "ERROR":
                self.high_issues.append(issue)
            elif severity == "WARNING":
                self.medium_issues.append(issue)
            else:
                self.low_issues.append(issue)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate security audit report."""
        total_issues = (
            len(self.critical_issues) +
            len(self.high_issues) +
            len(self.medium_issues) +
            len(self.low_issues)
        )
        
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "summary": {
                "total_issues": total_issues,
                "critical": len(self.critical_issues),
                "high": len(self.high_issues),
                "medium": len(self.medium_issues),
                "low": len(self.low_issues)
            },
            "tool_results": self.results,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if self.critical_issues:
            recommendations.append("CRITICAL: Address all critical security issues immediately")
        
        if self.high_issues:
            recommendations.append("HIGH: Address high-severity issues as soon as possible")
        
        if self.medium_issues:
            recommendations.append("MEDIUM: Plan to address medium-severity issues in next sprint")
        
        if self.low_issues:
            recommendations.append("LOW: Address low-severity issues during regular maintenance")
        
        # General recommendations
        recommendations.extend([
            "Regularly update dependencies to latest secure versions",
            "Implement automated security scanning in CI/CD pipeline",
            "Conduct regular security code reviews",
            "Use security-focused development practices",
            "Implement proper input validation and sanitization",
            "Use secure coding practices and avoid common vulnerabilities"
        ])
        
        return recommendations
    
    def save_report(self, output_file: str = "security_audit_report.json") -> None:
        """Save security audit report to file."""
        report = self._generate_report()
        
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security audit report saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print security audit summary."""
        report = self._generate_report()
        summary = report["summary"]
        
        print("\n" + "="*50)
        print("SECURITY AUDIT SUMMARY")
        print("="*50)
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Critical: {summary['critical']}")
        print(f"High: {summary['high']}")
        print(f"Medium: {summary['medium']}")
        print(f"Low: {summary['low']}")
        print("="*50)
        
        if summary['critical'] > 0:
            print("🚨 CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
        elif summary['high'] > 0:
            print("⚠️  HIGH SEVERITY ISSUES FOUND - PRIORITY ACTION REQUIRED")
        elif summary['medium'] > 0:
            print("ℹ️  MEDIUM SEVERITY ISSUES FOUND - PLAN TO ADDRESS")
        else:
            print("✅ NO CRITICAL OR HIGH SEVERITY ISSUES FOUND")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")


def main():
    """Main function for security audit."""
    # Set up logging
    setup_logging({
        'level': 'INFO',
        'enable_console': True,
        'enable_structured': True,
        'enable_colors': True
    })
    
    # Initialize auditor
    auditor = SecurityAuditor()
    
    # Run security audit
    report = auditor.run_security_audit()
    
    # Print summary
    auditor.print_summary()
    
    # Save report
    auditor.save_report()
    
    # Exit with appropriate code
    if report["summary"]["critical"] > 0:
        sys.exit(1)  # Critical issues found
    elif report["summary"]["high"] > 0:
        sys.exit(2)  # High issues found
    else:
        sys.exit(0)  # No critical or high issues


if __name__ == "__main__":
    main()

