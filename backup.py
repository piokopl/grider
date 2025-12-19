"""
Backup and Restore System
Export/import configuration and state
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import yaml
import zipfile
import io


@dataclass
class BackupMetadata:
    version: str = "1.0"
    created_at: str = ""
    bot_version: str = "1.0.0"
    pairs: List[str] = None
    total_grids: int = 0
    total_trades: int = 0
    total_pnl: float = 0
    notes: str = ""


class BackupManager:
    """
    Manages backup and restore of bot configuration and state
    """
    
    BACKUP_VERSION = "1.0"
    
    def __init__(self, logger: logging.Logger, 
                 config_dir: str = "configs",
                 db_path: str = "grid_bot.db",
                 backup_dir: str = "backups"):
        self.logger = logger
        self.config_dir = config_dir
        self.db_path = db_path
        self.backup_dir = backup_dir
        
        # Create backup directory if needed
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_backup(self, include_db: bool = True, 
                     include_configs: bool = True,
                     notes: str = "") -> str:
        """
        Create a full backup
        Returns: path to backup file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"grid_bot_backup_{timestamp}.zip"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Create metadata
                metadata = self._create_metadata(notes)
                zf.writestr("metadata.json", json.dumps(asdict(metadata), indent=2))
                
                # Backup configs
                if include_configs:
                    self._backup_configs(zf)
                
                # Backup main config
                if os.path.exists("config.yaml"):
                    zf.write("config.yaml", "config.yaml")
                
                # Backup database
                if include_db and os.path.exists(self.db_path):
                    zf.write(self.db_path, "grid_bot.db")
                
                # Backup paper trading state
                if os.path.exists("paper_state.json"):
                    zf.write("paper_state.json", "paper_state.json")
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            raise
    
    def restore_backup(self, backup_path: str, 
                      restore_db: bool = True,
                      restore_configs: bool = True) -> bool:
        """
        Restore from a backup file
        """
        if not os.path.exists(backup_path):
            self.logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zf:
                # Read metadata
                metadata_str = zf.read("metadata.json").decode('utf-8')
                metadata = json.loads(metadata_str)
                
                self.logger.info(f"Restoring backup from {metadata.get('created_at', 'unknown')}")
                self.logger.info(f"Pairs: {metadata.get('pairs', [])}")
                
                # Create backup of current state before restore
                self.create_backup(notes="pre_restore_backup")
                
                # Restore configs
                if restore_configs:
                    for name in zf.namelist():
                        if name.startswith("configs/") and name.endswith(".yaml"):
                            zf.extract(name, ".")
                    
                    if "config.yaml" in zf.namelist():
                        zf.extract("config.yaml", ".")
                
                # Restore database
                if restore_db and "grid_bot.db" in zf.namelist():
                    # Close any existing connections first
                    zf.extract("grid_bot.db", ".")
                
                # Restore paper state
                if "paper_state.json" in zf.namelist():
                    zf.extract("paper_state.json", ".")
            
            self.logger.info("Backup restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        
        for filename in os.listdir(self.backup_dir):
            if filename.endswith(".zip"):
                filepath = os.path.join(self.backup_dir, filename)
                try:
                    with zipfile.ZipFile(filepath, 'r') as zf:
                        metadata_str = zf.read("metadata.json").decode('utf-8')
                        metadata = json.loads(metadata_str)
                        
                        backups.append({
                            "filename": filename,
                            "path": filepath,
                            "created_at": metadata.get("created_at", ""),
                            "pairs": metadata.get("pairs", []),
                            "total_pnl": metadata.get("total_pnl", 0),
                            "notes": metadata.get("notes", ""),
                            "size_mb": os.path.getsize(filepath) / 1024 / 1024
                        })
                except:
                    # Invalid backup file
                    continue
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    def delete_backup(self, backup_path: str) -> bool:
        """Delete a backup file"""
        if os.path.exists(backup_path):
            os.remove(backup_path)
            self.logger.info(f"Deleted backup: {backup_path}")
            return True
        return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Keep only the most recent backups"""
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            to_delete = backups[keep_count:]
            for backup in to_delete:
                self.delete_backup(backup["path"])
            self.logger.info(f"Cleaned up {len(to_delete)} old backups")
    
    def _create_metadata(self, notes: str) -> BackupMetadata:
        """Create backup metadata"""
        pairs = []
        total_grids = 0
        total_trades = 0
        total_pnl = 0
        
        # Get info from database
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get pairs
                cursor.execute("SELECT DISTINCT pair FROM grids")
                pairs = [row[0] for row in cursor.fetchall()]
                
                # Get totals
                cursor.execute("SELECT COUNT(*) FROM grids")
                total_grids = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM grid_orders WHERE status = 'filled'")
                total_trades = cursor.fetchone()[0]
                
                cursor.execute("SELECT SUM(realized_pnl) FROM grids")
                result = cursor.fetchone()[0]
                total_pnl = result if result else 0
                
                conn.close()
            except:
                pass
        
        # Get pairs from configs
        if not pairs and os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".yaml"):
                    pair = filename.replace(".yaml", "")
                    pairs.append(pair)
        
        return BackupMetadata(
            version=self.BACKUP_VERSION,
            created_at=datetime.utcnow().isoformat(),
            pairs=pairs,
            total_grids=total_grids,
            total_trades=total_trades,
            total_pnl=total_pnl,
            notes=notes
        )
    
    def _backup_configs(self, zf: zipfile.ZipFile):
        """Backup all config files"""
        if os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".yaml"):
                    filepath = os.path.join(self.config_dir, filename)
                    zf.write(filepath, os.path.join("configs", filename))
    
    # === EXPORT/IMPORT SPECIFIC DATA ===
    
    def export_pair_config(self, pair: str) -> Dict:
        """Export configuration for a single pair"""
        config_path = os.path.join(self.config_dir, f"{pair}.yaml")
        
        if not os.path.exists(config_path):
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def import_pair_config(self, pair: str, config: Dict) -> bool:
        """Import configuration for a single pair"""
        config_path = os.path.join(self.config_dir, f"{pair}.yaml")
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Imported config for {pair}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to import config for {pair}: {e}")
            return False
    
    def export_statistics(self) -> Dict:
        """Export all statistics"""
        stats = {}
        
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get all grids
                cursor.execute("""
                    SELECT pair, grid_mode, total_trades, realized_pnl, created_at, stopped_at
                    FROM grids ORDER BY created_at
                """)
                
                stats["grids"] = [
                    {
                        "pair": row[0],
                        "mode": row[1],
                        "trades": row[2],
                        "pnl": row[3],
                        "created": row[4],
                        "stopped": row[5]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get summary by pair
                cursor.execute("""
                    SELECT pair, 
                           COUNT(*) as grid_count,
                           SUM(total_trades) as total_trades,
                           SUM(realized_pnl) as total_pnl
                    FROM grids GROUP BY pair
                """)
                
                stats["by_pair"] = {
                    row[0]: {
                        "grids": row[1],
                        "trades": row[2],
                        "pnl": row[3]
                    }
                    for row in cursor.fetchall()
                }
                
                conn.close()
            except Exception as e:
                self.logger.error(f"Failed to export statistics: {e}")
        
        return stats
    
    def export_trades_csv(self, filepath: str) -> bool:
        """Export all trades to CSV"""
        if not os.path.exists(self.db_path):
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT o.pair, o.order_type, o.side, o.price, o.amount, 
                       o.fill_price, o.fill_amount, o.fee, o.status, o.created_at, o.filled_at
                FROM grid_orders o
                WHERE o.status = 'filled'
                ORDER BY o.filled_at
            """)
            
            with open(filepath, 'w') as f:
                f.write("pair,type,side,price,amount,fill_price,fill_amount,fee,status,created,filled\n")
                for row in cursor.fetchall():
                    f.write(",".join(str(v) if v is not None else "" for v in row) + "\n")
            
            conn.close()
            self.logger.info(f"Exported trades to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
            return False


class ConfigHotReload:
    """
    Hot reload configuration without restarting bot
    """
    
    def __init__(self, config_dir: str, logger: logging.Logger):
        self.config_dir = config_dir
        self.logger = logger
        self._last_modified: Dict[str, float] = {}
        self._callbacks: Dict[str, List[callable]] = {}
    
    def register_callback(self, pair: str, callback: callable):
        """Register callback for config changes"""
        if pair not in self._callbacks:
            self._callbacks[pair] = []
        self._callbacks[pair].append(callback)
    
    def check_for_changes(self) -> List[str]:
        """Check for config file changes"""
        changed = []
        
        for filename in os.listdir(self.config_dir):
            if not filename.endswith(".yaml"):
                continue
            
            filepath = os.path.join(self.config_dir, filename)
            mtime = os.path.getmtime(filepath)
            
            if filepath in self._last_modified:
                if mtime > self._last_modified[filepath]:
                    changed.append(filename.replace(".yaml", ""))
            
            self._last_modified[filepath] = mtime
        
        return changed
    
    async def reload_config(self, pair: str) -> Optional[Dict]:
        """Reload config for a pair and notify callbacks"""
        config_path = os.path.join(self.config_dir, f"{pair}.yaml")
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Reloaded config for {pair}")
            
            # Notify callbacks
            if pair in self._callbacks:
                for callback in self._callbacks[pair]:
                    try:
                        callback(config)
                    except Exception as e:
                        self.logger.error(f"Config callback error: {e}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to reload config for {pair}: {e}")
            return None
