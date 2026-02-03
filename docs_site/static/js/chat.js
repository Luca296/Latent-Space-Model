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
    const newChatBtn = document.getElementById('new-chat');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const history = document.querySelector('.chat-history');
    const welcome = document.querySelector('.chatbot-welcome');
    const exampleBtns = document.querySelectorAll('.example-q');
    const askDocsBtn = document.getElementById('ask-docs-btn');

    // Open chatbot
    function openChatbot() {
        sidebar.classList.add('open');
        document.body.classList.add('chat-open');
        input.focus();
    }

    // Close chatbot
    function closeChatbot() {
        sidebar.classList.remove('open');
        document.body.classList.remove('chat-open');
    }

    // Toggle chatbot panel
    if (fab) fab.addEventListener('click', openChatbot);
    if (closeBtn) closeBtn.addEventListener('click', closeChatbot);
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function() {
            conversationHistory = [];
            if (history) history.innerHTML = '';
            if (welcome) welcome.style.display = 'block';
            if (input) input.value = '';
            openChatbot();
        });
    }

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
        const typingId = showTypingIndicator();
        sendBtn.disabled = true;

        let botMessageDiv = null;
        let botText = '';
        let botSources = [];

        try {
            const response = await fetch('/api/ask_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    history: conversationHistory
                })
            });

            if (!response.ok || !response.body) {
                removeTypingIndicator(typingId);
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                sendBtn.disabled = false;
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop() || '';

                for (const part of parts) {
                    const event = parseSSE(part);
                    if (!event) continue;

                    if (event.type === 'delta') {
                        if (typingId) removeTypingIndicator(typingId);
                        if (!botMessageDiv) {
                            botMessageDiv = addMessage('', 'bot');
                        }
                        botText += event.data.content || '';
                        updateMessage(botMessageDiv, botText, botSources);
                    }

                    if (event.type === 'sources') {
                        botSources = event.data.sources || [];
                        if (botMessageDiv) updateMessage(botMessageDiv, botText, botSources);
                    }

                    if (event.type === 'done') {
                        if (botMessageDiv) {
                            conversationHistory.push(
                                { role: 'user', content: question },
                                { role: 'assistant', content: botText }
                            );
                            if (conversationHistory.length > 8) {
                                conversationHistory = conversationHistory.slice(-8);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            addMessage('Sorry, I could not connect to the server. Please check your connection.', 'bot');
            console.error('Error:', error);
        }

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
        bubble.innerHTML = renderMarkdown(text);

        messageDiv.appendChild(bubble);

        if (sources && sources.length > 0 && sender === 'bot') {
            const sourcesDiv = buildSources(sources);
            bubble.appendChild(sourcesDiv);
        }

        history.appendChild(messageDiv);
        history.scrollTop = history.scrollHeight;

        return messageDiv;
    }

    function updateMessage(messageDiv, text, sources = []) {
        const bubble = messageDiv.querySelector('.message-bubble');
        if (!bubble) return;
        bubble.innerHTML = renderMarkdown(text);

        const existingSources = bubble.querySelector('.message-sources');
        if (existingSources) existingSources.remove();

        if (sources && sources.length > 0) {
            bubble.appendChild(buildSources(sources));
        }

        history.scrollTop = history.scrollHeight;
    }

    function buildSources(sources) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';

        const sourcesTitle = document.createElement('strong');
        sourcesTitle.textContent = 'Sources:';
        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach(source => {
            const link = document.createElement('a');
            link.className = 'source-link';
            const anchor = '#' + slugify(source.section || '');
            link.href = anchor;
            link.textContent = `[${source.id}] ${source.source} - ${source.section}`;
            link.addEventListener('click', function(e) {
                const targetId = anchor.replace('#', '');
                const target = document.getElementById(targetId);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
            sourcesDiv.appendChild(link);
        });

        return sourcesDiv;
    }

    function slugify(text) {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^a-z0-9\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-');
    }

    function renderMarkdown(text) {
        if (window.marked && window.DOMPurify) {
            const raw = window.marked.parse(text || '', { breaks: true, gfm: true });
            return window.DOMPurify.sanitize(raw);
        }
        return (text || '').replace(/[&<>"']/g, function(ch) {
            const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
            return map[ch] || ch;
        });
    }

    function parseSSE(block) {
        const lines = block.split('\n');
        let type = 'message';
        let data = '';

        for (const line of lines) {
            if (line.startsWith('event:')) {
                type = line.replace('event:', '').trim();
            } else if (line.startsWith('data:')) {
                data += line.replace('data:', '').trim();
            }
        }

        if (!data) return null;
        try {
            return { type, data: JSON.parse(data) };
        } catch {
            return null;
        }

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
