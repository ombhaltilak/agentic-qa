"""
Financial Document RAG — Built-in Demo System Under Test.

This is the DEFAULT SUT included for demonstration. It simulates
a financial document processing pipeline with intentional weaknesses.

Users can swap this out with their own RAG by using APIAdapter or
CallableAdapter instead.
"""

import re
import time
from agentic_qa.sut.base import BaseSUTAdapter


class FinancialDocumentRAG(BaseSUTAdapter):
    """
    Built-in demo: Financial Document Extraction system.
    
    Has intentional weaknesses for adversarial testing to discover:
    1. Poor handling of negative values
    2. No validation of impossible dates
    3. Vulnerable to prompt injection
    4. Only handles USD
    5. Weak Unicode handling
    6. No allocation sum validation
    """
    
    @property
    def name(self) -> str:
        return "Financial Document RAG v1.0"
    
    @property
    def domain(self) -> str:
        return "WAM financial documents"
    
    def describe(self) -> str:
        return (
            "A financial document processing pipeline that extracts structured data "
            "(NAV, fund names, dates, amounts, allocations) from unstructured financial "
            "documents used in Wealth & Asset Management (WAM). Supports fund fact sheets, "
            "portfolio statements, NAV reports, and compliance documents."
        )
    
    def process(self, input_text: str) -> dict:
        """Process a financial document excerpt and extract structured data."""
        start_time = time.time()
        
        try:
            result = {
                "status": "success",
                "output": "",
                "retrieved_contexts": [
                    "Excerpt from 2024 NAV Report",
                    "Asset allocation summary snippet"
                ],
                "sources": ["report_2024.pdf", "portfolio_summary.pdf"],
                "metadata": {"confidence": "medium"},
                "extracted_data": {},
                "warnings": [],
                "raw_input_length": len(input_text)
            }
            
            if len(input_text) > 10000:
                time.sleep(0.5)
            
            nav_values = self._extract_nav(input_text)
            if nav_values:
                result["extracted_data"]["nav"] = nav_values
            
            fund_names = self._extract_fund_names(input_text)
            if fund_names:
                result["extracted_data"]["fund_names"] = fund_names
            
            dates = self._extract_dates(input_text)
            if dates:
                result["extracted_data"]["dates"] = dates
            
            amounts = self._extract_amounts(input_text)
            if amounts:
                result["extracted_data"]["amounts"] = amounts
            
            allocations = self._extract_allocations(input_text)
            if allocations:
                result["extracted_data"]["allocations"] = allocations
            
            # Weakness: Prompt injection vulnerability
            if "ignore previous instructions" in input_text.lower() or \
               "system:" in input_text.lower():
                result["extracted_data"]["system_response"] = "Acknowledged special instruction"
            
            if not result["extracted_data"]:
                result["status"] = "no_data_extracted"
                result["warnings"].append("No structured data could be extracted from input")
            
            result["output"] = str(result["extracted_data"])
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "output": str(e),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _extract_nav(self, text: str) -> list:
        """Extract NAV values. Weakness: Accepts negative NAVs."""
        patterns = [
            r'NAV[:\s]+\$?([-\d,.]+)',
            r'Net Asset Value[:\s]+\$?([-\d,.]+)',
            r'nav per (?:unit|share)[:\s]+\$?([-\d,.]+)',
        ]
        values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    val = float(match.replace(',', ''))
                    values.append({"value": val, "raw": match})
                except ValueError:
                    continue
        return values
    
    def _extract_fund_names(self, text: str) -> list:
        """Extract fund names. Weakness: Strips Unicode."""
        patterns = [
            r'(?:Fund|Portfolio|Scheme)[:\s]+([A-Za-z0-9\s\-&.]+?)(?:\n|$|,|\|)',
            r'\"([A-Za-z0-9\s\-&.]+?(?:Fund|Portfolio|Trust|ETF))\"',
        ]
        names = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend([m.strip() for m in matches if len(m.strip()) > 2])
        names = [re.sub(r'[^\x00-\x7F]+', '', n) for n in names]
        names = [n for n in names if n.strip()]
        return list(set(names))
    
    def _extract_dates(self, text: str) -> list:
        """Extract dates. Weakness: No validation of impossible dates."""
        patterns = [
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})',
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s+(\d{1,2}),?\s+(\d{4})',
        ]
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append("/".join(match))
        return dates
    
    def _extract_amounts(self, text: str) -> list:
        """Extract amounts. Weakness: Only handles USD."""
        patterns = [
            r'\$\s*([-\d,.]+(?:\.\d{1,2})?)\s*(?:million|billion|M|B|K)?',
            r'USD\s*([-\d,.]+(?:\.\d{1,2})?)',
        ]
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    val = float(match.replace(',', ''))
                    amounts.append({"value": val, "raw": f"${match}"})
                except ValueError:
                    continue
        return amounts
    
    def _extract_allocations(self, text: str) -> list:
        """Extract allocations. Weakness: No sum validation."""
        pattern = r'([A-Za-z\s]+?):\s*(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pattern, text)
        allocations = []
        for name, pct in matches:
            allocations.append({
                "asset_class": name.strip(),
                "percentage": float(pct)
            })
        return allocations


# Backward compatibility
sut_instance = FinancialDocumentRAG()

def get_sut() -> FinancialDocumentRAG:
    return sut_instance

def get_sut_description() -> str:
    return sut_instance.describe()
