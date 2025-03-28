import redis
import uuid
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MAX_USERS = int(os.getenv("MAX_USERS", 10))
TTL_SECONDS = int(os.getenv("TTL_SECONDS", 3600))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

def get_current_users() -> int:
    current = redis_client.get("flow:1:current_users")
    current_value = int(current) if current else 0
    print(f"[DEBUG] get_current_users: current_users = {current_value}")
    return current_value

def increment_users() -> str:
    current_users = get_current_users()
    print(f"[DEBUG] increment_users: Checking limit, current_users = {current_users}, MAX_USERS = {MAX_USERS}")
    
    if current_users >= MAX_USERS:
        print(f"[DEBUG] increment_users: Max users exceeded, returning None")
        return None
    
    request_id = str(uuid.uuid4())
    user_key = f"flow:1:user:{request_id}"
    print(f"[DEBUG] increment_users: Generated request_id = {request_id}, user_key = {user_key}")
    
    redis_client.incr("flow:1:current_users")
    redis_client.setex(user_key, TTL_SECONDS, "active")
    
    new_count = get_current_users()
    print(f"[DEBUG] increment_users: Incremented count, new_count = {new_count}, TTL set to {TTL_SECONDS}s")
    return request_id

def decrement_users():
    current = get_current_users()
    print(f"[DEBUG] decrement_users: Starting with current_users = {current}")
    
    if current <= 0:
        print(f"[DEBUG] decrement_users: No users to decrement, exiting")
        return
    
    expired = 0
    for key in redis_client.scan_iter("flow:1:user:*"):
        if not redis_client.exists(key):
            expired += 1
            print(f"[DEBUG] decrement_users: Found expired key = {key}")
    
    if expired > 0:
        redis_client.decr("flow:1:current_users", expired)
        print(f"[DEBUG] decrement_users: Decreased count by {expired} expired keys, new_count = {get_current_users()}")
    
    if get_current_users() > 0:
        redis_client.decr("flow:1:current_users")
        print(f"[DEBUG] decrement_users: Decreased count by 1, final_count = {get_current_users()}")
    else:
        print(f"[DEBUG] decrement_users: Count already at 0 after expired keys, no further decrement")