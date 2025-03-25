import time

class APIManager:
    def __init__(self):
        # Initialize rate limit tracking
        self.request_timestamps = []
        self.rate_limit_window = 60  # 60 seconds window
        self.max_requests = 100      # Maximum requests per window
        
        # API key management
        self.api_keys = []
        self.current_key_index = 0
        self.last_rotation_time = time.time()
        self.rotation_interval = 3600  # Rotate keys every hour

    def handle_rate_limits(self):
        """
        Handle API rate limits by implementing a sliding window rate limiter.
        Returns True if request can proceed, False if rate limit exceeded.
        """
        current_time = time.time()
        
        # Remove timestamps outside the window
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts <= self.rate_limit_window
        ]
        
        # Check if we're within rate limits
        if len(self.request_timestamps) >= self.max_requests:
            return False
        
        # Add current timestamp and allow request
        self.request_timestamps.append(current_time)
        return True
        
    def manage_api_keys(self):
        """
        Manage and rotate API keys to distribute load and handle key rotation.
        Returns the current active API key.
        """
        current_time = time.time()
        
        # Check if it's time to rotate keys
        if current_time - self.last_rotation_time >= self.rotation_interval:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.last_rotation_time = current_time
            
        # Return current active key if available
        if self.api_keys:
            return self.api_keys[self.current_key_index]
        else:
            raise ValueError("No API keys configured")

    def add_api_key(self, key):
        """Add a new API key to the rotation pool"""
        if key not in self.api_keys:
            self.api_keys.append(key)

    def remove_api_key(self, key):
        """Remove an API key from the rotation pool"""
        if key in self.api_keys:
            self.api_keys.remove(key)
            # Adjust current index if necessary
            if self.current_key_index >= len(self.api_keys):
                self.current_key_index = 0 