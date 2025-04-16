from typing import Dict, List, Optional
import re
from datetime import datetime


class ResponseValidator:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.min_content_length = 50
        self.max_content_length = 1000

    def validate_source(self, source: Dict) -> Dict:
        """Validate a single source result"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "confidence": source.get("confidence", 0),
        }

        # Check confidence score
        if source.get("confidence", 0) < self.confidence_threshold:
            validation["warnings"].append("Low confidence score")

        # Check content length
        content = source.get("content", "")
        if len(content) < self.min_content_length:
            validation["warnings"].append("Content too short")
        elif len(content) > self.max_content_length:
            validation["warnings"].append("Content too long")

        # Check for potential hallucination indicators
        if self._check_hallucination_indicators(content):
            validation["warnings"].append("Potential hallucination detected")
            validation["is_valid"] = False

        # Validate metadata
        if not self._validate_metadata(source):
            validation["warnings"].append("Invalid or missing metadata")
            validation["is_valid"] = False

        return validation

    def _check_hallucination_indicators(self, content: str) -> bool:
        """Check for common hallucination indicators in content"""
        indicators = [
            r"I think",
            r"probably",
            r"might be",
            r"could be",
            r"I believe",
            r"perhaps",
            r"maybe",
        ]

        # Count uncertain language
        uncertainty_count = sum(
            1
            for indicator in indicators
            if re.search(indicator, content, re.IGNORECASE)
        )
        return uncertainty_count >= 2

    def _validate_metadata(self, source: Dict) -> bool:
        """Validate source metadata"""
        required_fields = ["document_id", "page_number"]
        return all(field in source for field in required_fields)

    def validate_query_response(self, response: Dict) -> Dict:
        """Validate entire query response"""
        validation = {"is_valid": True, "warnings": [], "source_validations": []}

        if "error" in response:
            validation["is_valid"] = False
            validation["warnings"].append(f"Query error: {response['error']}")
            return validation

        sources = response.get("sources", [])
        if not sources:
            validation["warnings"].append("No sources found")
            validation["is_valid"] = False
            return validation

        # Validate each source
        for source in sources:
            source_validation = self.validate_source(source)
            validation["source_validations"].append(source_validation)

            if not source_validation["is_valid"]:
                validation["is_valid"] = False

            validation["warnings"].extend(source_validation["warnings"])

        # Check for duplicate content
        if self._check_duplicates(sources):
            validation["warnings"].append("Duplicate content detected")

        return validation

    def _check_duplicates(self, sources: List[Dict]) -> bool:
        """Check for duplicate content in sources"""
        contents = [source.get("content", "") for source in sources]
        return len(contents) != len(set(contents))


def format_validation_results(validation: Dict) -> str:
    """Format validation results for display"""
    output = []

    if not validation["is_valid"]:
        output.append("⚠️ Response contains validation issues:")
    else:
        output.append("✅ Response passed validation")

    if validation["warnings"]:
        output.append("\nWarnings:")
        for warning in validation["warnings"]:
            output.append(f"- {warning}")

    if "source_validations" in validation:
        output.append("\nSource Validations:")
        for idx, source_val in enumerate(validation["source_validations"], 1):
            output.append(f"\nSource {idx}:")
            output.append(f"- Confidence: {source_val['confidence']:.2f}")
            if source_val["warnings"]:
                output.append("- Warnings:")
                for warning in source_val["warnings"]:
                    output.append(f"  • {warning}")

    return "\n".join(output)
