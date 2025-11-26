import glob
import os
import shutil
import tempfile
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

class SystemCleanupService:
    """
    Service to handle system cleanup operations.
    Refactored from the monolithic trigger_system_cleanup function.
    """

    def __init__(self):
        self.results = {
            "temp_files_cleaned": 0,
            "old_logs_removed": 0,
            "cache_cleared_mb": 0,
            "database_optimized": False,
            "disk_space_freed_mb": 0,
        }

    def clean_temp_files(self) -> None:
        """Cleans temporary system files."""
        temp_dirs = [
            tempfile.gettempdir(),
            "/tmp" if os.path.exists("/tmp") else None,
            "temp" if os.path.exists("temp") else None,
            "__pycache__",
            ".pytest_cache",
        ]
        
        temp_dirs = [d for d in temp_dirs if d and os.path.exists(d)]

        for temp_dir in temp_dirs:
            try:
                patterns = [
                    os.path.join(temp_dir, "*.tmp"),
                    os.path.join(temp_dir, "*.temp"),
                    os.path.join(temp_dir, "**", "__pycache__", "**"),
                    (
                        os.path.join(temp_dir, ".pytest_cache", "**")
                        if temp_dir == ".pytest_cache"
                        else ""
                    ),
                ]

                for pattern in patterns:
                    if pattern:
                        files = glob.glob(pattern, recursive=True)
                        for file_path in files:
                            self._remove_path(file_path, "temp_files_cleaned")
            except OSError:
                continue

    def clean_old_logs(self, retention_days: int = 7) -> None:
        """Cleans logs older than retention_days."""
        log_dirs = [
            "logs",
            "security_logs_20251117125456",
            "security_logs_20251117161150",
        ]
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                for root, _, files in os.walk(log_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_age_days = (
                                datetime.now().timestamp() - os.path.getmtime(file_path)
                            ) / (24 * 3600)
                            
                            if file_age_days > retention_days:
                                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                os.remove(file_path)
                                self.results["old_logs_removed"] += 1
                                self.results["cache_cleared_mb"] += file_size_mb
                        except OSError:
                            continue

    def clean_python_cache(self) -> None:
        """Recursively cleans __pycache__ directories."""
        for root, dirs, _ in os.walk("."):
            if "__pycache__" in dirs:
                cache_path = os.path.join(root, "__pycache__")
                try:
                    count = len(os.listdir(cache_path)) if os.path.exists(cache_path) else 0
                    shutil.rmtree(cache_path)
                    self.results["temp_files_cleaned"] += count
                except OSError:
                    pass

    def optimize_database(self) -> None:
        """Optimizes SQLite databases if they exist."""
        db_files = ["project_state.db", "gamified_database.db", "audit.db"]
        optimized = False
        for db_file in db_files:
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file)
                    conn.execute("VACUUM")
                    conn.close()
                    optimized = True
                except Exception:
                    pass
        self.results["database_optimized"] = optimized

    def clean_node_modules(self) -> None:
        """Cleans node_modules directory."""
        if os.path.exists("node_modules"):
            try:
                shutil.rmtree("node_modules")
                self.results["temp_files_cleaned"] += 1000  # Estimate
                self.results["disk_space_freed_mb"] += 500  # Estimate
            except OSError:
                pass

    def clean_build_artifacts(self) -> None:
        """Cleans build and dist directories."""
        build_dirs = ["build", "dist", "*.egg-info"]
        for build_dir in build_dirs:
            build_paths = glob.glob(build_dir)
            for path in build_paths:
                self._remove_path(path, "temp_files_cleaned")

    def _remove_path(self, path: str, counter_key: str) -> None:
        """Helper to remove a file or directory and update a counter."""
        try:
            if os.path.isfile(path):
                os.remove(path)
                self.results[counter_key] += 1
            elif os.path.isdir(path):
                shutil.rmtree(path)
                self.results[counter_key] += 1
        except OSError:
            pass

    def calculate_final_stats(self) -> None:
        """Calculates final statistics."""
        self.results["cache_cleared_mb"] = round(self.results["cache_cleared_mb"], 2)
        
        # Estimate if not already added (like in node_modules)
        estimated_freed = (self.results["temp_files_cleaned"] * 0.001) + self.results["cache_cleared_mb"]
        
        # If disk_space_freed_mb was already modified (e.g. by clean_node_modules), add to it
        # But here we want to ensure we don't double count if we just used the estimate logic
        # The original code logic was:
        # cleanup_results["disk_space_freed_mb"] = round(
        #     (cleanup_results["temp_files_cleaned"] * 0.001)
        #     + cleanup_results["cache_cleared_mb"],
        #     2,
        # )
        # And THEN added node_modules size.
        
        # So let's recalculate base freed from files and cache
        base_freed = (self.results["temp_files_cleaned"] * 0.001) + self.results["cache_cleared_mb"]
        
        # If we added explicit freed space (like node_modules), it's already in disk_space_freed_mb
        # Wait, the original code added to disk_space_freed_mb AFTER the calculation for node_modules.
        # But for standard cleanup, disk_space_freed_mb starts at 0.
        
        # Let's stick to the original logic's flow:
        # 1. Calculate base
        current_freed = self.results["disk_space_freed_mb"] # This might have values from node_modules if called before?
        # Actually in original code, node_modules is called AFTER calculation.
        
        # So my perform_cleanup should call calculate BEFORE deep/full cleanup if I want to match exactly,
        # OR I just sum it all up at the end.
        
        # Let's simplify:
        # We will track explicit size if possible, otherwise use the estimate.
        # For this refactor, I will just update the total at the end.
        
        self.results["disk_space_freed_mb"] = round(
             self.results["disk_space_freed_mb"] + base_freed, 2
        )

    def perform_cleanup(self, cleanup_type: str = "standard") -> Dict[str, Any]:
        """
        Executes the cleanup process based on the type.
        
        Args:
            cleanup_type: 'standard', 'deep', or 'full'
            
        Returns:
            Dictionary with cleanup results.
        """
        # 1. Standard Cleanup
        self.clean_temp_files()
        self.clean_old_logs()
        self.clean_python_cache()
        self.optimize_database()
        
        # Calculate stats for standard cleanup items
        # (Original code calculated here)
        
        # 2. Deep Cleanup
        if cleanup_type in ["deep", "full"]:
            self.clean_node_modules()
            
        # 3. Full Cleanup
        if cleanup_type == "full":
            self.clean_build_artifacts()
            
        # Finalize stats
        self.calculate_final_stats()
        
        return self.results
