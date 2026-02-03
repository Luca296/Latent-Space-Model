/**
 * Chatbot and UI interactions for the Latent-Space-Model docs site.
 */

// Conversation history for context
let conversationHistory = [];

/**
 * Initialize all UI components when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    initThemeToggle();
    initChatbot();
    initNavigation();
    initTOC();
    initSearch();
});

/**
 * Theme toggle functionality
 */
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    const html = document.documentElement;

    // Load saved theme or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);

    toggle.addEventListener('click', function() {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
}

/**
 * Chatbot functionality
 */
function initChatbot() {
    const fab = document.getElementById('chatbot-toggle');
    const sidebar = document.getElementById('chatbot-sidebar');
    const closeBtn = document.getElementById('chatbot-close');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const history = document.querySelector('.chat-history');
    const welcome = document.querySelector('.chatbot-welcome');
    const exampleBtns = document.querySelectorAll('.example-q');
    const askDocsBtn = document.getElementById('ask-docs-btn');

    // Open chatbot
    function openChatbot() {
        sidebar.classList.add('open');
        input.focus();
    }

    // Close chatbot
    function closeChatbot() {
        sidebar.classList.remove('open');
    }

    // Toggle chatbot panel
    if (fab) fab.addEventListener('click', openChatbot);
    if (closeBtn) closeBtn.addEventListener('click', closeChatbot);

    // Open chatbot from hero button
    if (askDocsBtn) {
        askDocsBtn.addEventListener('click', function(e) {
            e.preventDefault();
            openChatbot();
        });
    }

    // Example question buttons
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const question = this.textContent;
            welcome.style.display = 'none';
            addMessage(question, 'user');
            sendQuestion(question);
        });
    });

    // Send message on submit
    function sendMessage() {
        const question = input.value.trim();
        if (!question) return;

        // Hide welcome message on first question
        if (welcome) welcome.style.display = 'none';

        // Add user message
        addMessage(question, 'user');

        // Clear input
        input.value = '';

        // Send to API
        sendQuestion(question);
    }

    // Send question to API
    async function sendQuestion(question) {
        // Show typing indicator
        const typingId = showTypingIndicator();

        // Disable send button
        sendBtn.disabled = true;

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    history: conversationHistory
                })
            });

            // Remove typing indicator
            removeTypingIndicator(typingId);

            if (response.ok) {
                const data = await response.json();

                // Add bot response
                addMessage(data.answer, 'bot', data.sources);

                // Update conversation history
                conversationHistory.push(
                    { role: 'user', content: question },
                    { role: 'assistant', content: data.answer }
                );

                // Keep only last 4 exchanges (8 messages)
                if (conversationHistory.length > 8) {
                    conversationHistory = conversationHistory.slice(-8);
                }
            } else {
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            addMessage('Sorry, I could not connect to the server. Please check your connection.', 'bot');
            console.error('Error:', error);
        }

        // Re-enable send button
        sendBtn.disabled = false;
    }

    // Send on button click
    sendBtn.addEventListener('click', sendMessage);

    // Send on Enter key
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Add message to chat
    function addMessage(text, sender, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;

        messageDiv.appendChild(bubble);

        // Add sources if available
        if (sources && sources.length > 0 && sender === 'bot') {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';

            const sourcesTitle = document.createElement('strong');
            sourcesTitle.textContent = 'Sources:';
            sourcesDiv.appendChild(sourcesTitle);

            sources.forEach(source => {
                const link = document.createElement('a');
                link.className = 'source-link';
                link.href = '#';
                link.textContent = `[${source.id}] ${source.source} - ${source.section}`;
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    // Could scroll to source anchor or open source viewer
                });
                sourcesDiv.appendChild(link);
            });

            messageDiv.appendChild(sourcesDiv);
        }

        history.appendChild(messageDiv);
        history.scrollTop = history.scrollHeight;
    }

    // Show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot';
        typingDiv.id = 'typing-' + Date.now();

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator message-bubble';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicator.appendChild(dot);
        }

        typingDiv.appendChild(indicator);
        history.appendChild(typingDiv);
        history.scrollTop = history.scrollHeight;

        return typingDiv.id;
    }

    // Remove typing indicator
    function removeTypingIndicator(id) {
        const indicator = document.getElementById(id);
        if (indicator) {
            indicator.remove();
        }
    }
}

/**
 * Navigation toggle functionality
 */
function initNavigation() {
    const navToggles = document.querySelectorAll('.nav-toggle');

    navToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const submenu = this.nextElementSibling;
            const isExpanded = submenu.classList.contains('expanded');

            // Close all other sections (optional - for accordion behavior)
            // document.querySelectorAll('.nav-submenu').forEach(menu => {
            //     menu.classList.remove('expanded');
            // });
            // document.querySelectorAll('.nav-toggle').forEach(t => {
            //     t.classList.remove('active');
            // });

            // Toggle current section
            if (isExpanded) {
                submenu.classList.remove('expanded');
                this.classList.remove('active');
            } else {
                submenu.classList.add('expanded');
                this.classList.add('active');
            }
        });
    });

    // Highlight current page in nav
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-submenu a').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

/**
 * Table of Contents generation
 */
function initTOC() {
    const tocList = document.getElementById('toc-list');
    const headings = document.querySelectorAll('.content h2, .content h3');

    if (!tocList || headings.length === 0) return;

    headings.forEach(heading => {
        if (!heading.id) return;

        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#' + heading.id;
        a.textContent = heading.textContent;
        a.dataset.target = heading.id;

        // Indent h3
        if (heading.tagName === 'H3') {
            li.style.paddingLeft = '12px';
        }

        a.addEventListener('click', function(e) {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth' });
        });

        li.appendChild(a);
        tocList.appendChild(li);
    });

    // Update active TOC link on scroll
    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -80% 0px',
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                document.querySelectorAll('.toc a').forEach(link => {
                    link.classList.remove('active');
                });
                const activeLink = document.querySelector(`.toc a[data-target="${id}"]`);
                if (activeLink) {
                    activeLink.classList.add('active');
                }
            }
        });
    }, observerOptions);

    headings.forEach(heading => {
        if (heading.id) {
            observer.observe(heading);
        }
    });
}

/**
 * Search functionality
 */
function initSearch() {
    const searchBox = document.getElementById('search-box');
    if (!searchBox) return;

    // Simple in-page search highlighting
    searchBox.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        const content = document.querySelector('.content');

        if (!query) {
            // Remove all highlights
            content.querySelectorAll('.search-highlight').forEach(el => {
                const parent = el.parentNode;
                parent.replaceChild(document.createTextNode(el.textContent), el);
                parent.normalize();
            });
            return;
        }

        // For a real implementation, you would index content and show results
        // This is a placeholder for the visual demo
    });

    // Allow submitting search
    searchBox.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const query = this.value;
            if (query) {
                // Could trigger search API or scroll to first match
                console.log('Search:', query);
            }
        }
    });
}

/**
 * Utility: Debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Utility: Copy code to clipboard
 */
function initCodeCopy() {
    document.querySelectorAll('pre code').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 8px;
            font-size: 0.75rem;
            background: var(--color-sidebar-bg);
            border: 1px solid var(--color-border);
            border-radius: 4px;
            cursor: pointer;
            color: var(--color-text);
        `;

        button.addEventListener('click', async function() {
            try {
                await navigator.clipboard.writeText(block.textContent);
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = 'Copy';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });

        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
}

// Initialize code copy on load
document.addEventListener('DOMContentLoaded', initCodeCopy);
