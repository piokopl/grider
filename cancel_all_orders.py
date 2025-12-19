#!/usr/bin/env python3
"""
Cancel all open orders on Bitget and reset database.
Use this to clean up after bugs.
"""

import asyncio
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchange import BitgetExchange
from database import Database


async def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize exchange
    exchange = BitgetExchange(
        api_key=config.get("api_key", config.get("api", {}).get("key", "")),
        api_secret=config.get("api_secret", config.get("api", {}).get("secret", "")),
        passphrase=config.get("passphrase", config.get("api", {}).get("passphrase", "")),
        paper_trading=config.get("paper_trading", False)
    )
    
    # Initialize database
    db = Database()
    
    print("=" * 60)
    print("CANCEL ALL ORDERS SCRIPT")
    print("=" * 60)
    
    # Get all pairs from configs directory
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    pairs = []
    
    if os.path.exists(configs_dir):
        for filename in os.listdir(configs_dir):
            if filename.endswith(".yaml"):
                pair_config_path = os.path.join(configs_dir, filename)
                with open(pair_config_path) as f:
                    pair_config = yaml.safe_load(f)
                    if pair_config.get("enabled", True):
                        pair = pair_config.get("pair")
                        if pair:
                            pairs.append(pair)
    
    # Fallback to old config format
    if not pairs:
        pairs = [p["pair"] for p in config.get("pairs", []) if p.get("enabled", True)]
    
    print(f"\nPairs to process: {len(pairs)}")
    
    total_cancelled = 0
    
    for pair in pairs:
        print(f"\n{pair}:")
        
        try:
            # Get open orders from exchange
            orders = await exchange.get_open_orders(pair)
            
            if orders:
                print(f"  Found {len(orders)} open orders")
                
                # Cancel each order
                for order in orders:
                    # OrderResult is a dataclass, use attributes not .get()
                    order_id = order.order_id if hasattr(order, 'order_id') else order.get("orderId")
                    if order_id:
                        try:
                            await exchange.cancel_order(pair, order_id)
                            print(f"    Cancelled: {order_id}")
                            total_cancelled += 1
                        except Exception as e:
                            print(f"    Failed to cancel {order_id}: {e}")
                        
                        await asyncio.sleep(0.1)  # Rate limit
            else:
                print(f"  No open orders")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Total cancelled: {total_cancelled} orders")
    
    # Reset database
    print("\nResetting database...")
    
    # Mark all pending orders as cancelled
    try:
        conn = db._get_conn()
        cursor = conn.cursor()
        
        # Update grid_levels
        cursor.execute("""
            UPDATE grid_levels 
            SET buy_status = 'none', buy_order_id = NULL
            WHERE buy_status = 'pending'
        """)
        buy_reset = cursor.rowcount
        
        cursor.execute("""
            UPDATE grid_levels 
            SET sell_status = 'none', sell_order_id = NULL
            WHERE sell_status = 'pending'
        """)
        sell_reset = cursor.rowcount
        
        # Update orders table
        cursor.execute("""
            UPDATE orders 
            SET status = 'cancelled'
            WHERE status = 'pending'
        """)
        orders_reset = cursor.rowcount
        
        conn.commit()
        
        print(f"  Reset {buy_reset} buy levels")
        print(f"  Reset {sell_reset} sell levels")
        print(f"  Reset {orders_reset} orders")
        
    except Exception as e:
        print(f"  Database error: {e}")
    
    print("\n✅ Done! You can now restart the bot with a clean state.")
    print("   The bot will place only 6 orders per pair (3 BUY + 3 SELL)")


if __name__ == "__main__":
    asyncio.run(main())
