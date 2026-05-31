#!/usr/bin/env python3
"""
Clear Redis cache utility.

Usage:
    python clear_cache.py              # Clear all caches
    python clear_cache.py --user usr_a1b2c3d4  # Clear specific user
    python clear_cache.py --responses  # Clear only response cache
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env", override=False)

from src.config import Config
from src.redis_cache import RedisSemanticCache


def main():
    parser = argparse.ArgumentParser(description="Clear Redis cache")
    parser.add_argument("--user", help="Clear cache for specific user ID")
    parser.add_argument("--responses", action="store_true", help="Clear only response cache")
    parser.add_argument("--all", action="store_true", help="Clear all caches (default)")
    args = parser.parse_args()

    # Create Redis cache instance
    cache = RedisSemanticCache(
        redis_url=Config.REDIS_URL,
        response_ttl=Config.RESPONSE_CACHE_TTL,
    )

    if not cache.available:
        print("❌ Redis is not available. Check your REDIS_URL configuration.")
        print(f"   Current REDIS_URL: {Config.REDIS_URL}")
        return 1

    print(f"✓ Connected to Redis: {Config.REDIS_URL}")
    print()

    # Clear based on arguments
    if args.user:
        print(f"Clearing cache for user: {args.user}")
        deleted = cache.invalidate_user_responses(args.user)
        print(f"✓ Deleted {deleted} cached response(s)")
        
    elif args.responses:
        print("Clearing response cache only...")
        deleted = 0
        try:
            for key in cache._client.scan_iter("resp:*", count=500):
                cache._client.delete(key)
                deleted += 1
            print(f"✓ Deleted {deleted} cached response(s)")
        except Exception as exc:
            print(f"❌ Error: {exc}")
            return 1
            
    else:  # --all or default
        print("Clearing ALL caches (responses, profiles, history, viz state)...")
        result = cache.clear_all_caches()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return 1
        
        print(f"✓ Cleared caches:")
        print(f"  - Responses:    {result['responses']}")
        print(f"  - Profiles:     {result['profiles']}")
        print(f"  - History:      {result['history']}")
        print(f"  - Viz State:    {result['viz_state']}")
        print(f"  - Total:        {result['total']} keys")

    print()
    print("✓ Cache cleared successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
