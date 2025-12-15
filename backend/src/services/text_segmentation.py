import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextSegmentationService:
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def segment_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Segment a large text into smaller chunks suitable for embedding
        """
        if not text:
            return []

        # Clean and normalize the text
        cleaned_text = self._clean_text(text)

        # Split text into sentences
        sentences = self._split_into_sentences(cleaned_text)

        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences)

        # Create segment objects with metadata
        segments = []
        for i, chunk in enumerate(chunks):
            segment = {
                "content": chunk,
                "metadata": {
                    **(metadata or {}),
                    "segment_id": i,
                    "total_segments": len(chunks)
                }
            }
            segments.append(segment)

        logger.info(f"Segmented text into {len(segments)} chunks")
        return segments

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Pattern to match sentence endings (., !, ?) followed by whitespace or end of string
        sentence_pattern = r'[.!?]+\s+|\.+$'
        sentences = re.split(sentence_pattern, text)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into chunks that don't exceed the maximum size
        """
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed the max size
            if len(current_chunk + " " + sentence) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # If the current chunk is not empty, save it
                if current_chunk:
                    chunks.append(current_chunk)

                # If the sentence is longer than max_chunk_size, split it
                if len(sentence) > self.max_chunk_size:
                    sub_chunks = self._split_large_sentence(sentence)
                    chunks.extend(sub_chunks[:-1])  # Add all but the last chunk
                    current_chunk = sub_chunks[-1]  # Keep the last chunk as current
                else:
                    current_chunk = sentence  # Start a new chunk with this sentence

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_large_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence that is too large into smaller chunks
        """
        if len(sentence) <= self.max_chunk_size:
            return [sentence]

        chunks = []
        start = 0

        while start < len(sentence):
            end = start + self.max_chunk_size

            # If we're not at the end, try to break at a word boundary
            if end < len(sentence):
                # Find the last space within the chunk
                space_index = sentence.rfind(' ', start, end)
                if space_index != -1 and space_index > start:
                    end = space_index

            chunk = sentence[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position - include overlap if possible
            if end > start:
                start = end
                # Add overlap by going back overlap_size characters (but not before the end of the current chunk)
                if start > self.overlap_size:
                    start = max(start - self.overlap_size, end)
            else:
                # If we can't advance, move forward by one character to avoid infinite loop
                start += 1

        return chunks

# Global instance
text_segmentation_service = TextSegmentationService()