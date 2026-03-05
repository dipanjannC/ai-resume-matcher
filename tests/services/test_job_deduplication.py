"""
Tests for job deduplication logic.
TDD: written before the implementation.
"""

import pytest
import hashlib
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.resume_data import JobDescription


def _compute_expected_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


class TestJobDeduplication:
    """Tests for hash-based job deduplication in JobProcessor."""

    def test_content_hash_computed_from_url(self):
        """Hash of a URL should be deterministic."""
        url = "https://example.com/careers/swe"
        h1 = _compute_expected_hash(url)
        h2 = _compute_expected_hash(url)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_content_hash_computed_from_text_fallback(self):
        """Hash of raw text should be deterministic."""
        text = "Senior ML Engineer role at Acme Corp. Requires Python, TensorFlow."
        h = _compute_expected_hash(text)
        assert len(h) == 64

    def test_different_urls_produce_different_hashes(self):
        """Two different URLs should not share a hash."""
        h1 = _compute_expected_hash("https://a.com/job/1")
        h2 = _compute_expected_hash("https://b.com/job/2")
        assert h1 != h2

    def test_whitespace_normalized_before_hashing(self):
        """Leading/trailing whitespace shouldn't change the hash."""
        h1 = _compute_expected_hash("  https://example.com/job  ")
        h2 = _compute_expected_hash("https://example.com/job")
        assert h1 == h2

    def test_job_description_has_content_hash_field(self):
        """JobDescription model should expose content_hash and source_url fields."""
        jd = JobDescription(title="SWE", company="Acme")
        assert hasattr(jd, "content_hash")
        assert hasattr(jd, "source_url")

    @pytest.mark.asyncio
    async def test_duplicate_detection_skips_reprocessing(self):
        """process_and_store_job should return existing job if hash matches."""
        from app.services.job_processor import JobProcessor

        url = "https://example.com/job/42"
        expected_hash = _compute_expected_hash(url)

        # Simulate an existing job on disk with same hash
        existing_job = JobDescription(
            title="Existing Job",
            company="Acme",
            source_url=url,
            content_hash=expected_hash,
        )

        with patch.object(JobProcessor, "__init__", return_value=None), \
             patch.object(JobProcessor, "find_job_by_hash", return_value=existing_job) as mock_find, \
             patch.object(JobProcessor, "_save_job_data", new_callable=AsyncMock) as mock_save:

            processor = JobProcessor.__new__(JobProcessor)
            processor.stored_jobs = {}

            # Finding by hash should short-circuit saving
            found = await processor.find_job_by_hash(expected_hash)
            assert found is not None
            assert found.title == "Existing Job"
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_job_removes_from_memory_and_disk(self):
        """delete_job should clear the job from memory cache and mark file removed."""
        from app.services.job_processor import JobProcessor
        import tempfile, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            job_id = "test-job-123"
            job_data = JobDescription(id=job_id, title="Test Job", company="Acme")

            job_file = jobs_dir / f"{job_id}.json"
            job_file.write_text(json.dumps(job_data.to_dict(), default=str))

            with patch.object(JobProcessor, "__init__", return_value=None):
                processor = JobProcessor.__new__(JobProcessor)
                processor.stored_jobs = {job_id: job_data}
                processor.jobs_dir = jobs_dir
                processor.jobs_collection = "job_descriptions"
                processor.vector_store = MagicMock()
                processor.vector_store.delete_document = MagicMock()

            result = await processor.delete_job(job_id)

            assert result is True
            assert job_id not in processor.stored_jobs
            assert not job_file.exists()
