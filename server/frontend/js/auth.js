/**
 * Authentication Manager
 * Handles login, signup, and user authentication including OAuth
 */

class AuthManager {
    constructor() {
        console.log('🏗️ AuthManager constructor called');
        this.apiBase = '';
        this.token = localStorage.getItem('authToken');
        this.user = null;
        
        console.log('💾 Token from localStorage:', this.token ? `${this.token.substring(0, 20)}...` : 'None');
        
        // Load user data if token exists
        if (this.token) {
            console.log('🔄 Token found, loading user data...');
            this.loadUserData();
        } else {
            console.log('⚠️ No token found in localStorage');
        }
    }

    /**
     * Initialize login form
     */
    static initLogin() {
        console.log('🔧 Initializing login form...');
        const authManager = new AuthManager();
        const form = document.getElementById('loginForm');
        const usernameInput = document.getElementById('username');
        const passwordInput = document.getElementById('password');
        const passwordToggle = document.getElementById('passwordToggle');
        const submitBtn = form.querySelector('.auth-btn');
        const rememberCheckbox = document.getElementById('rememberMe');
        const googleBtn = document.querySelector('.google-btn');

        console.log('🎯 Form elements found:', {
            form: !!form,
            usernameInput: !!usernameInput,
            passwordInput: !!passwordInput,
            googleBtn: !!googleBtn
        });

        // Password visibility toggle
        if (passwordToggle) {
            passwordToggle.addEventListener('click', () => {
                console.log('👁️ Password visibility toggled');
                authManager.togglePasswordVisibility(passwordInput, passwordToggle);
            });
        }

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('📝 Login form submitted');
            await authManager.handleLogin(form, submitBtn);
        });

        // Google OAuth login
        if (googleBtn) {
            googleBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                console.log('🔵 Google OAuth button clicked');
                await authManager.handleGoogleLogin();
            });
        }

        // Real-time validation
        usernameInput.addEventListener('input', () => {
            authManager.validateField(usernameInput, authManager.validateUsername);
        });

        passwordInput.addEventListener('input', () => {
            authManager.validateField(passwordInput, authManager.validatePassword);
        });

        // Auto-fill from localStorage if remember me was checked
        const rememberedUsername = localStorage.getItem('rememberedUsername');
        if (rememberedUsername) {
            console.log('💭 Auto-filling remembered username:', rememberedUsername);
            usernameInput.value = rememberedUsername;
            rememberCheckbox.checked = true;
        }

        // Handle OAuth callback if present
        authManager.handleOAuthCallback();
        console.log('✅ Login form initialization complete');
    }

    /**
     * Initialize signup form
     */
    static initSignup() {
        console.log('🔧 Initializing signup form...');
        const authManager = new AuthManager();
        const form = document.getElementById('signupForm');
        const usernameInput = document.getElementById('username');
        const emailInput = document.getElementById('email');
        const passwordInput = document.getElementById('password');
        const confirmPasswordInput = document.getElementById('confirmPassword');
        const passwordToggle = document.getElementById('passwordToggle');
        const confirmPasswordToggle = document.getElementById('confirmPasswordToggle');
        const submitBtn = form.querySelector('.auth-btn');
        const googleBtn = document.querySelector('.google-btn');

        console.log('🎯 Signup form elements found:', {
            form: !!form,
            usernameInput: !!usernameInput,
            emailInput: !!emailInput,
            googleBtn: !!googleBtn
        });

        // Password visibility toggles
        if (passwordToggle) {
            passwordToggle.addEventListener('click', () => {
                console.log('👁️ Password visibility toggled');
                authManager.togglePasswordVisibility(passwordInput, passwordToggle);
            });
        }

        if (confirmPasswordToggle) {
            confirmPasswordToggle.addEventListener('click', () => {
                console.log('👁️ Confirm password visibility toggled');
                authManager.togglePasswordVisibility(confirmPasswordInput, confirmPasswordToggle);
            });
        }

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('📝 Signup form submitted');
            await authManager.handleSignup(form, submitBtn);
        });

        // Google OAuth signup
        if (googleBtn) {
            googleBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                console.log('🔵 Google OAuth signup button clicked');
                await authManager.handleGoogleLogin();
            });
        }

        // Real-time validation
        usernameInput.addEventListener('input', () => {
            authManager.validateField(usernameInput, authManager.validateUsername);
        });

        emailInput.addEventListener('input', () => {
            authManager.validateField(emailInput, authManager.validateEmail);
        });

        passwordInput.addEventListener('input', () => {
            authManager.validateField(passwordInput, authManager.validatePassword);
            // Revalidate confirm password if it has content
            if (confirmPasswordInput.value) {
                authManager.validatePasswordMatch(passwordInput, confirmPasswordInput);
            }
        });

        confirmPasswordInput.addEventListener('input', () => {
            authManager.validatePasswordMatch(passwordInput, confirmPasswordInput);
        });

        // Handle OAuth callback if present
        authManager.handleOAuthCallback();
        console.log('✅ Signup form initialization complete');
    }

    /**
     * Handle Google OAuth login
     */
    async handleGoogleLogin() {
        try {
            console.log('🔵 Starting Google OAuth flow...');
            // Get Google OAuth URL
            const response = await fetch('/auth/oauth/google');
            const data = await response.json();

            console.log('📡 OAuth URL response:', {
                status: response.status,
                ok: response.ok,
                hasAuthUrl: !!data.auth_url,
                hasState: !!data.state
            });

            if (response.ok) {
                console.log('💾 Storing OAuth state for verification...');
                // Store state for verification
                localStorage.setItem('oauth_state', data.state);
                
                console.log('🔄 Redirecting to Google OAuth...');
                // Redirect to Google OAuth
                window.location.href = data.auth_url;
            } else {
                console.error('❌ OAuth URL request failed:', data);
                this.showError('OAuth not configured. Please contact administrator.');
            }
        } catch (error) {
            console.error('💥 Google OAuth error:', error);
            this.showError('Failed to initiate Google login. Please try again.');
        }
    }

    /**
     * Handle OAuth callback (when returning from Google)
     */
    async handleOAuthCallback() {
        console.log('🔍 Checking for OAuth callback parameters...');
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');
        const state = urlParams.get('state');
        const error = urlParams.get('error');

        console.log('📋 OAuth callback parameters:', {
            code: code ? `${code.substring(0, 20)}...` : null,
            state: state ? `${state.substring(0, 20)}...` : null,
            error: error,
            fullURL: window.location.href
        });

        // Check for OAuth errors in URL
        if (error) {
            console.error('❌ OAuth error in URL:', error);
            // Clear URL parameters
            window.history.replaceState({}, document.title, window.location.pathname);
            this.showError(this.getOAuthErrorMessage(error));
            return;
        }

        // Check if this is an OAuth callback
        if (!code) {
            console.log('ℹ️ No OAuth code found, not an OAuth callback');
            return;
        }

        console.log('🔄 Processing OAuth callback...');
        // Clear URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);

        if (code && state) {
            try {
                // Verify state matches what we stored
                const storedState = localStorage.getItem('oauth_state');
                console.log('🔐 Verifying OAuth state...', {
                    receivedState: state ? `${state.substring(0, 20)}...` : null,
                    storedState: storedState ? `${storedState.substring(0, 20)}...` : null,
                    matches: state === storedState
                });

                if (state !== storedState) {
                    console.error('❌ OAuth state mismatch!');
                    this.showError('OAuth security error. Please try again.');
                    return;
                }

                // Clear stored state
                localStorage.removeItem('oauth_state');
                console.log('🧹 Cleared stored OAuth state');

                // Note: The callback is now handled by the server redirect
                // This code shouldn't typically run since we redirect to oauth-success.html
                console.warn('⚠️ OAuth callback reached client-side handler - this should redirect server-side');
                this.showError('OAuth callback handling error. Please try signing in again.');

            } catch (error) {
                console.error('💥 OAuth callback error:', error);
                this.showError('Failed to complete OAuth authentication.');
            }
        }
    }

    /**
     * Get user-friendly OAuth error message
     */
    getOAuthErrorMessage(error) {
        console.log('📝 Getting OAuth error message for:', error);
        switch (error) {
            case 'incomplete_user_info':
                return 'We couldn\'t get your complete profile information from Google. Please try again.';
            case 'oauth_failed':
                return 'OAuth authentication failed. Please try signing in again.';
            default:
                return `Authentication error: ${error}. Please try again.`;
        }
    }

    /**
     * Check for OAuth errors on page load
     */
    static checkOAuthErrors() {
        console.log('🔍 Checking for OAuth errors in URL...');
        const urlParams = new URLSearchParams(window.location.search);
        const error = urlParams.get('error');
        
        if (error) {
            console.error('❌ OAuth error found in URL:', error);
            const authManager = new AuthManager();
            // Clear URL parameters
            window.history.replaceState({}, document.title, window.location.pathname);
            authManager.showError(authManager.getOAuthErrorMessage(error));
        } else {
            console.log('✅ No OAuth errors found');
        }
    }

    /**
     * Handle login form submission
     */
    async handleLogin(form, submitBtn) {
        try {
            console.log('🔐 Processing login...');
            this.setLoading(submitBtn, true);
            this.hideMessages();

            const formData = new FormData(form);
            const userData = {
                username: formData.get('username'),
                password: formData.get('password')
            };

            console.log('📤 Sending login request for user:', userData.username);

            const response = await fetch('/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userData)
            });

            const data = await response.json();
            console.log('📡 Login response:', {
                status: response.status,
                ok: response.ok,
                hasToken: !!data.access_token,
                hasUser: !!data.user
            });

            if (response.ok) {
                console.log('✅ Login successful! User:', {
                    id: data.user.id,
                    username: data.user.username,
                    email: data.user.email,
                    oauth_provider: data.user.oauth_provider
                });

                // Store token and user data
                this.token = data.access_token;
                this.user = data.user;
                localStorage.setItem('authToken', this.token);
                console.log('💾 Token and user data stored');
                
                // Handle remember me
                const rememberMe = form.querySelector('#rememberMe').checked;
                if (rememberMe) {
                    console.log('💭 Saving username for next time');
                    localStorage.setItem('rememberedUsername', userData.username);
                } else {
                    localStorage.removeItem('rememberedUsername');
                }

                this.showSuccess('Login successful! Redirecting...');
                
                console.log('🔄 Redirecting to main application in 1.5 seconds...');
                // Redirect to main application
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            } else {
                console.error('❌ Login failed:', data.detail || 'Unknown error');
                this.showError(data.detail || 'Login failed. Please try again.');
            }
        } catch (error) {
            console.error('💥 Login network error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.setLoading(submitBtn, false);
        }
    }

    /**
     * Handle signup form submission
     */
    async handleSignup(form, submitBtn) {
        try {
            this.setLoading(submitBtn, true);
            this.hideMessages();

            // Validate form before submission
            if (!this.validateSignupForm(form)) {
                this.setLoading(submitBtn, false);
                return;
            }

            const formData = new FormData(form);
            const userData = {
                username: formData.get('username'),
                email: formData.get('email'),
                password: formData.get('password'),
                confirm_password: formData.get('confirm_password')
            };

            const response = await fetch('/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userData)
            });

            const data = await response.json();

            if (response.ok) {
                // Store token and user data
                this.token = data.access_token;
                this.user = data.user;
                localStorage.setItem('authToken', this.token);

                this.showSuccess('Account created successfully! Redirecting...');
                
                // Redirect to main application
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            } else {
                // Handle validation errors
                if (data.detail && Array.isArray(data.detail)) {
                    const errorMessages = data.detail.map(err => err.msg).join(', ');
                    this.showError(errorMessages);
                } else {
                    this.showError(data.detail || 'Registration failed. Please try again.');
                }
            }
        } catch (error) {
            console.error('Signup error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.setLoading(submitBtn, false);
        }
    }

    /**
     * Toggle password visibility
     */
    togglePasswordVisibility(input, toggle) {
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        
        // Update eye icon
        const eyeIcon = toggle.querySelector('.eye-icon');
        if (isPassword) {
            // Change to "eye-off" icon
            eyeIcon.innerHTML = `
                <path d="m1 1 22 22"></path>
                <path d="M6.71 6.71C2.5 9.7 1 12 1 12s4 8 11 8c1.59 0 3.09-.29 4.47-.77L6.71 6.71Z"></path>
                <path d="M10.9 4.24C11.25 4.08 11.62 4 12 4c7 0 11 8 11 8a20.65 20.65 0 0 1-2.26 3.07l-9.83-9.83Z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            `;
        } else {
            // Change back to "eye" icon
            eyeIcon.innerHTML = `
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            `;
        }
    }

    /**
     * Field validation
     */
    validateField(input, validator) {
        const isValid = validator.call(this, input.value);
        this.updateFieldValidation(input, isValid);
        return isValid;
    }

    /**
     * Validate username
     */
    validateUsername(username) {
        if (!username || username.length < 3) return false;
        if (username.length > 50) return false;
        return /^[a-zA-Z0-9_]+$/.test(username);
    }

    /**
     * Validate email
     */
    validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Validate password
     */
    validatePassword(password) {
        return password && password.length >= 6;
    }

    /**
     * Validate password match
     */
    validatePasswordMatch(passwordInput, confirmPasswordInput) {
        const isMatch = passwordInput.value === confirmPasswordInput.value;
        this.updateFieldValidation(confirmPasswordInput, isMatch);
        return isMatch;
    }

    /**
     * Update field validation UI
     */
    updateFieldValidation(input, isValid) {
        const container = input.closest('.input-container');
        const line = container.querySelector('.input-line');
        
        if (input.value === '') {
            // No validation for empty fields
            container.classList.remove('error', 'success');
            return;
        }

        if (isValid) {
            container.classList.remove('error');
            container.classList.add('success');
            line.style.background = 'linear-gradient(45deg, #4ecdc4, #44a08d)';
        } else {
            container.classList.remove('success');
            container.classList.add('error');
            line.style.background = 'linear-gradient(45deg, #ff6b6b, #ff5252)';
        }
    }

    /**
     * Validate entire signup form
     */
    validateSignupForm(form) {
        const username = form.querySelector('#username').value;
        const email = form.querySelector('#email').value;
        const password = form.querySelector('#password').value;
        const confirmPassword = form.querySelector('#confirmPassword').value;

        let isValid = true;

        if (!this.validateUsername(username)) {
            this.showError('Username must be 3-50 characters and contain only letters, numbers, and underscores.');
            isValid = false;
        }

        if (!this.validateEmail(email)) {
            this.showError('Please enter a valid email address.');
            isValid = false;
        }

        if (!this.validatePassword(password)) {
            this.showError('Password must be at least 6 characters long.');
            isValid = false;
        }

        if (password !== confirmPassword) {
            this.showError('Passwords do not match.');
            isValid = false;
        }
        return isValid;
    }

    /**
     * Set loading state for button
     */
    setLoading(button, isLoading) {
        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');

        if (isLoading) {
            btnText.style.display = 'none';
            btnLoader.style.display = 'flex';
            button.disabled = true;
        } else {
            btnText.style.display = 'block';
            btnLoader.style.display = 'none';
            button.disabled = false;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        const successElement = document.getElementById('successMessage');
        
        if (successElement) successElement.style.display = 'none';
        
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const errorElement = document.getElementById('errorMessage');
        const successElement = document.getElementById('successMessage');
        
        if (errorElement) errorElement.style.display = 'none';
        
        if (successElement) {
            successElement.textContent = message;
            successElement.style.display = 'block';
        }
    }

    /**
     * Hide all messages
     */
    hideMessages() {
        const errorElement = document.getElementById('errorMessage');
        const successElement = document.getElementById('successMessage');
        
        if (errorElement) errorElement.style.display = 'none';
        if (successElement) successElement.style.display = 'none';
    }

    /**
     * Load user data from server
     */
    async loadUserData() {
        try {
            console.log('📡 Loading user data from server...');
            const response = await fetch('/auth/me', {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            console.log('📡 User data response:', {
                status: response.status,
                ok: response.ok,
                statusText: response.statusText
            });

            if (response.ok) {
                this.user = await response.json();
                console.log('✅ User data loaded successfully:', {
                    id: this.user.id,
                    username: this.user.username,
                    email: this.user.email,
                    oauth_provider: this.user.oauth_provider,
                    avatar_url: this.user.avatar_url ? 'Present' : 'None'
                });
            } else {
                console.warn('❌ Token validation failed, clearing auth data');
                // Token is invalid, remove it
                localStorage.removeItem('authToken');
                this.token = null;
                this.user = null;
            }
        } catch (error) {
            console.error('💥 Error loading user data:', error);
        }
    }

    /**
     * Logout user
     */
    async logout() {
        try {
            console.log('🚪 Logging out user...');
            await fetch('/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });
            console.log('📡 Logout request sent to server');
        } catch (error) {
            console.error('💥 Logout server error:', error);
        } finally {
            console.log('🧹 Clearing local authentication data...');
            // Clear local data regardless of server response
            localStorage.removeItem('authToken');
            localStorage.removeItem('oauth_state');
            this.token = null;
            this.user = null;
            
            console.log('🔄 Redirecting to login page...');
            // Redirect to login
            window.location.href = '/frontend/html/login.html';
        }
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated() {
        return !!this.token && !!this.user;
    }

    /**
     * Get current user
     */
    getCurrentUser() {
        return this.user;
    }

    /**
     * Get auth token
     */
    getToken() {
        return this.token;
    }
}

// CSS for validation states
const validationStyles = `
<style>
.input-container.success .input-line {
    background: linear-gradient(45deg, #4ecdc4, #44a08d) !important;
}

.input-container.error .input-line {
    background: linear-gradient(45deg, #ff6b6b, #ff5252) !important;
}

.input-container.success input {
    border-bottom-color: #4ecdc4 !important;
}

.input-container.error input {
    border-bottom-color: #ff6b6b !important;
}

.input-container.success label {
    color: #4ecdc4 !important;
}

.input-container.error label {
    color: #ff6b6b !important;
}

/* Fix for password toggle in validation states */
.input-container.success .password-toggle {
    color: rgba(255, 255, 255, 0.8) !important;
}

.input-container.error .password-toggle {
    color: rgba(255, 255, 255, 0.8) !important;
}

.input-container.success .password-toggle:hover,
.input-container.error .password-toggle:hover {
    color: #fff !important;
}

/* Ensure consistent positioning */
.input-container.success .password-toggle,
.input-container.error .password-toggle {
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 10;
}
</style>
`;

// Inject validation styles
document.head.insertAdjacentHTML('beforeend', validationStyles);

// Export for use in other scripts
window.AuthManager = AuthManager; 