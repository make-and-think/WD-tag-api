import redis

try:
    # Connect to KeyDB (or Redis)
    # By default, it connects to localhost on port 6379
    # You can specify host, port, and password if needed:
    # r = redis.Redis(host='your_keydb_host', port=6379, password='your_password')
    r = redis.Redis(decode_responses=True)  # decode_responses=True to get strings instead of bytes

    print("Successfully connected to KeyDB!")

    # --- String Operations ---
    # Set a key-value pair
    r.set('mykey', 'Hello KeyDB!')
    print(f"SET mykey: Hello KeyDB!")

    # Get the value of a key
    value = r.get('mykey')
    print(f"GET mykey: {value}")

    # Set a key with an expiration time (in seconds)
    r.setex('tempkey', 60, 'This key will expire in 60 seconds')
    print(f"SETEX tempkey (60s): This key will expire in 60 seconds")

    # --- Hash Operations ---
    # Set multiple fields in a hash
    r.hset('user:1000', mapping={
        'username': 'testuser',
        'email': 'testuser@example.com',
        'score': 100
    })
    print(f"HSET user:1000: username, email, score")

    # Get a specific field from a hash
    username = r.hget('user:1000', 'username')
    print(f"HGET user:1000 username: {username}")

    # Get all fields and values from a hash
    user_data = r.hgetall('user:1000')
    print(f"HGETALL user:1000: {user_data}")

    # --- List Operations ---
    # Push items to the end of a list (rpush)
    r.rpush('mylist', 'item1', 'item2', 'item3')
    print(f"RPUSH mylist: item1, item2, item3")

    # Get a range of items from a list
    list_items = r.lrange('mylist', 0, -1)  # Get all items
    print(f"LRANGE mylist 0 -1: {list_items}")

    # Pop an item from the beginning of a list (lpop)
    first_item = r.lpop('mylist')
    print(f"LPOP mylist: {first_item}")
    print(f"LRANGE mylist 0 -1 (after LPOP): {r.lrange('mylist', 0, -1)}")

    # --- Set Operations ---
    # Add members to a set
    r.sadd('myset', 'apple', 'banana', 'cherry', 'apple') # 'apple' will only be added once
    print(f"SADD myset: apple, banana, cherry, apple")

    # Get all members of a set
    set_members = r.smembers('myset')
    print(f"SMEMBERS myset: {set_members}")

    # Check if a member exists in a set
    is_member = r.sismember('myset', 'banana')
    print(f"SISMEMBER myset banana: {is_member}")

    # --- Other useful commands ---
    # Check if a key exists
    exists = r.exists('mykey', 'nonexistentkey')
    print(f"EXISTS mykey, nonexistentkey: {exists}") # Returns the number of existing keys

    # Delete a key
    deleted_count = r.delete('mykey', 'tempkey')
    print(f"DEL mykey, tempkey: {deleted_count} keys deleted")

    # Get all keys matching a pattern (use with caution in production on large databases)
    # SCAN is preferred for production environments [6]
    all_keys = []
    for key in r.scan_iter("*"): # [6]
        all_keys.append(key)
    print(f"SCAN *: Found keys: {all_keys}")


except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to KeyDB: {e}")

except Exception as e:
    print(f"An error occurred: {e}")