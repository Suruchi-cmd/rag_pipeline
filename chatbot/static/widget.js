/**
 * AeroBot Chat Widget
 *
 * Self-contained script that injects the chat widget UI into any page.
 * Embed options:
 *   1. Script tag:  <script src="https://your-domain/static/widget.js"></script>
 *   2. iframe:      <iframe src="https://your-domain/widget" width="400" height="600" />
 *
 * Colours: #F00C74 (pink), #39FF14 (green), #1a1a2e (dark)
 */
(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // Config
  // -------------------------------------------------------------------------

  // When loaded via <script> from a different origin, set this to the API
  // server's origin (e.g. "https://bot.aerosportsparks.ca").
  // Leave empty to use the same origin as the page.
  var API_BASE = (typeof AEROBOT_API_BASE !== 'undefined') ? AEROBOT_API_BASE : '';

  var WELCOME_MSG =
    "Hi! I'm AeroBot \uD83C\uDF89 Ask me anything about AeroSports Scarborough — " +
    "jump prices, birthday parties, group events, camp programs, and more!";

  var SESSION_KEY = 'aerobot_session_id';

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------

  var isOpen = false;
  var isStreaming = false;

  // -------------------------------------------------------------------------
  // Session management
  // -------------------------------------------------------------------------

  function getSessionId() {
    var id = sessionStorage.getItem(SESSION_KEY);
    if (!id) {
      id = generateUUID();
      sessionStorage.setItem(SESSION_KEY, id);
    }
    return id;
  }

  function setSessionId(id) {
    sessionStorage.setItem(SESSION_KEY, id);
  }

  function generateUUID() {
    if (crypto && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    // Fallback for older browsers
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0;
      var v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  // -------------------------------------------------------------------------
  // Bootstrap — inject CSS link + HTML
  // -------------------------------------------------------------------------

  function injectCSS() {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = API_BASE + '/static/widget.css';
    document.head.appendChild(link);
  }

  function buildHTML() {
    return (
      '<button id="aerobot-toggle" aria-label="Open AeroBot chat">' +
        '<div id="aerobot-badge"></div>' +
        '<svg id="aerobot-icon-chat" xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="white">' +
          '<path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>' +
        '</svg>' +
        '<svg id="aerobot-icon-close" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="white" style="display:none">' +
          '<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>' +
        '</svg>' +
      '</button>' +

      '<div id="aerobot-window" class="aerobot-closed">' +
        '<div id="aerobot-header">' +
          '<div id="aerobot-header-info">' +
            '<div id="aerobot-avatar">\uD83C\uDF89</div>' +
            '<div>' +
              '<div id="aerobot-title">AeroBot</div>' +
              '<div id="aerobot-subtitle">AeroSports Scarborough</div>' +
            '</div>' +
          '</div>' +
          '<button id="aerobot-close" aria-label="Close chat">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white">' +
              '<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>' +
            '</svg>' +
          '</button>' +
        '</div>' +

        '<div id="aerobot-messages"></div>' +

        '<div id="aerobot-input-area">' +
          '<input type="text" id="aerobot-input"' +
            ' placeholder="Ask about prices, parties, camps\u2026"' +
            ' maxlength="500"' +
            ' autocomplete="off"' +
            ' aria-label="Chat message"' +
          '/>' +
          '<button id="aerobot-send" aria-label="Send">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="white">' +
              '<path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>' +
            '</svg>' +
          '</button>' +
        '</div>' +

        '<div id="aerobot-footer">Powered by AeroSports \u2736</div>' +
      '</div>'
    );
  }

  function init() {
    injectCSS();

    var container = document.createElement('div');
    container.id = 'aerobot-container';
    container.innerHTML = buildHTML();
    document.body.appendChild(container);

    // Bind events
    document.getElementById('aerobot-toggle').addEventListener('click', toggleWidget);
    document.getElementById('aerobot-close').addEventListener('click', closeWidget);
    document.getElementById('aerobot-send').addEventListener('click', handleSendClick);
    document.getElementById('aerobot-input').addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendClick();
      }
    });
  }

  // -------------------------------------------------------------------------
  // Open / Close
  // -------------------------------------------------------------------------

  function toggleWidget() {
    if (isOpen) {
      closeWidget();
    } else {
      openWidget();
    }
  }

  function openWidget() {
    isOpen = true;
    document.getElementById('aerobot-window').classList.remove('aerobot-closed');
    document.getElementById('aerobot-toggle').classList.add('aerobot-open');
    document.getElementById('aerobot-icon-chat').style.display = 'none';
    document.getElementById('aerobot-icon-close').style.display = 'block';
    document.getElementById('aerobot-toggle').classList.remove('has-notification');

    // Show welcome on first open
    var messages = document.getElementById('aerobot-messages');
    if (messages.children.length === 0) {
      appendBotMessage(WELCOME_MSG);
    }

    setTimeout(function () {
      document.getElementById('aerobot-input').focus();
    }, 320);
  }

  function closeWidget() {
    isOpen = false;
    document.getElementById('aerobot-window').classList.add('aerobot-closed');
    document.getElementById('aerobot-toggle').classList.remove('aerobot-open');
    document.getElementById('aerobot-icon-chat').style.display = 'block';
    document.getElementById('aerobot-icon-close').style.display = 'none';
  }

  // -------------------------------------------------------------------------
  // Send
  // -------------------------------------------------------------------------

  function handleSendClick() {
    if (isStreaming) return;
    var input = document.getElementById('aerobot-input');
    var text = input.value.trim();
    if (!text) return;
    input.value = '';
    setSendDisabled(true);
    sendMessage(text);
  }

  function setSendDisabled(disabled) {
    document.getElementById('aerobot-send').disabled = disabled;
  }

  function sendMessage(text) {
    isStreaming = true;
    appendUserMessage(text);
    showTypingIndicator();

    var sessionId = getSessionId();
    var url = API_BASE + '/api/chat/stream' +
      '?message=' + encodeURIComponent(text) +
      '&session_id=' + encodeURIComponent(sessionId);

    var eventSource = new EventSource(url);
    var botMsgDiv = null;
    var fullText = '';

    eventSource.onmessage = function (e) {
      try {
        var data = JSON.parse(e.data);

        if (data.done) {
          eventSource.close();
          isStreaming = false;
          setSendDisabled(false);
          if (data.session_id) {
            setSessionId(data.session_id);
          }
          // Ensure typing indicator is gone
          hideTypingIndicator();
          return;
        }

        if (data.token) {
          if (!botMsgDiv) {
            hideTypingIndicator();
            botMsgDiv = createBotMessageDiv();
          }
          fullText += data.token;
          botMsgDiv.innerHTML = renderMarkdown(fullText);
          scrollToBottom();
        }
      } catch (err) {
        console.error('[AeroBot] parse error:', err);
      }
    };

    eventSource.onerror = function () {
      eventSource.close();
      isStreaming = false;
      setSendDisabled(false);
      hideTypingIndicator();
      if (!botMsgDiv) {
        appendBotMessage(
          'Sorry, I had trouble connecting right now. ' +
          'Please try again or call us at **289-454-5555**.'
        );
      }
    };
  }

  // -------------------------------------------------------------------------
  // Message helpers
  // -------------------------------------------------------------------------

  function appendUserMessage(text) {
    var div = document.createElement('div');
    div.className = 'aerobot-msg aerobot-msg-user';
    div.textContent = text;
    document.getElementById('aerobot-messages').appendChild(div);
    scrollToBottom();
  }

  function appendBotMessage(markdown) {
    var div = createBotMessageDiv();
    div.innerHTML = renderMarkdown(markdown);
    scrollToBottom();
    return div;
  }

  function createBotMessageDiv() {
    var div = document.createElement('div');
    div.className = 'aerobot-msg aerobot-msg-bot';
    document.getElementById('aerobot-messages').appendChild(div);
    return div;
  }

  function showTypingIndicator() {
    if (document.getElementById('aerobot-typing')) return;
    var div = document.createElement('div');
    div.id = 'aerobot-typing';
    div.className = 'aerobot-msg aerobot-msg-bot aerobot-typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    document.getElementById('aerobot-messages').appendChild(div);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    var el = document.getElementById('aerobot-typing');
    if (el) el.parentNode.removeChild(el);
  }

  function scrollToBottom() {
    var messages = document.getElementById('aerobot-messages');
    messages.scrollTop = messages.scrollHeight;
  }

  // -------------------------------------------------------------------------
  // Markdown renderer (bold + links + line-breaks)
  // -------------------------------------------------------------------------

  function renderMarkdown(text) {
    // 1. Escape HTML entities to prevent XSS
    var escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // 2. Bold: **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 3. Links: [label](https://...)
    escaped = escaped.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // 4. Bare URLs (not already inside an <a>)
    escaped = escaped.replace(
      /(?<![">])(https?:\/\/[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // 5. Line breaks
    escaped = escaped.replace(/\n/g, '<br>');

    return escaped;
  }

  // -------------------------------------------------------------------------
  // Auto-init
  // -------------------------------------------------------------------------

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
