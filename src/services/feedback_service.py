import logging
from typing import Dict, Any, List
from datetime import datetime
from src.models.schemas import FeedbackRequest
import json

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for handling user feedback and metrics"""
    
    def __init__(self):
        # For MVP, store feedback in memory
        # In production, this would use a proper database
        self.feedback_storage: List[Dict[str, Any]] = []
    
    async def store_feedback(self, feedback: FeedbackRequest) -> bool:
        """Store user feedback"""
        try:
            feedback_data = {
                "trace_id": feedback.trace_id,
                "rating": feedback.rating,
                "reason": feedback.reason,
                "proposed_answer": feedback.proposed_answer,
                "selected_citations": feedback.selected_citations,
                "tags": feedback.tags,
                "timestamp": datetime.now().isoformat(),
                "user_id": "anonymous"  # TODO: Extract from auth context
            }
            
            self.feedback_storage.append(feedback_data)
            logger.info(f"Feedback stored for trace_id: {feedback.trace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Feedback storage failed: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get feedback and performance metrics"""
        try:
            if not self.feedback_storage:
                return {
                    "positive_rate": 0.0,
                    "counts_by_reason": {},
                    "ndcg_5": None,
                    "mrr_5": None,
                    "trending_queries": []
                }
            
            # Calculate positive rate
            positive_count = sum(1 for f in self.feedback_storage if f["rating"] == "up")
            total_count = len(self.feedback_storage)
            positive_rate = (positive_count / total_count) * 100 if total_count > 0 else 0
            
            # Count reasons for negative feedback
            reason_counts = {}
            for feedback in self.feedback_storage:
                if feedback["rating"] == "down" and feedback["reason"]:
                    reason = feedback["reason"]
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            return {
                "positive_rate": round(positive_rate, 2),
                "counts_by_reason": reason_counts,
                "total_feedback": total_count,
                "positive_feedback": positive_count,
                "negative_feedback": total_count - positive_count,
                "ndcg_5": None,  # Would calculate from evaluation data
                "mrr_5": None,   # Would calculate from evaluation data
                "trending_queries": []  # Would analyze query patterns
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                "positive_rate": 0.0,
                "counts_by_reason": {},
                "error": str(e)
            }
    
    async def get_feedback_by_trace_id(self, trace_id: str) -> Dict[str, Any]:
        """Get feedback for specific trace ID"""
        try:
            for feedback in self.feedback_storage:
                if feedback["trace_id"] == trace_id:
                    return feedback
            return {}
            
        except Exception as e:
            logger.error(f"Feedback retrieval failed: {e}")
            return {}
    
    async def export_feedback(self) -> List[Dict[str, Any]]:
        """Export all feedback data"""
        return self.feedback_storage.copy()