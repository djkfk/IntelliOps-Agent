// IntelliOps Agent 前端应用
class IntelliOpsAgentApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:9900/api';
        this.currentMode = 'quick'; // 'quick' 或 'stream'
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.currentChatHistory = []; // 当前对话的消息历史
        this.chatHistories = this.loadChatHistories(); // 所有历史对话
        this.isCurrentChatFromHistory = false; // 标记当前对话是否是从历史记录加载的
        this._isAIOpsSession = false; // 当前是否为 AIOps 诊断会话
        this._aiopsHistorySaved = false; // AIOps 结果是否已写入侧边栏历史
        this._contextMaxTokens = 24000;
        this._contextBaseTokens = 0;
        this._contextTagTimer = null;
        
        this.initializeElements();
        this.bindEvents();
        this.initShellChrome();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
        this.refreshContextMeterFromLocal();
    }

    // 初始化Markdown配置
    initMarkdown() {
        // 等待 marked 库加载完成
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                try {
                    // 配置marked选项
                    marked.setOptions({
                        breaks: true,  // 支持GFM换行
                        gfm: true,     // 启用GitHub风格的Markdown
                        headerIds: false,
                        mangle: false
                    });

                    // 配置代码高亮
                    if (typeof hljs !== 'undefined') {
                        marked.setOptions({
                            highlight: function(code, lang) {
                                if (lang && hljs.getLanguage(lang)) {
                                    try {
                                        return hljs.highlight(code, { language: lang }).value;
                                    } catch (err) {
                                        console.error('代码高亮失败:', err);
                                    }
                                }
                                return code;
                            }
                        });
                    }
                    console.log('Markdown 渲染库初始化成功');
                } catch (e) {
                    console.error('Markdown 配置失败:', e);
                }
            } else {
                // 如果 marked 还没加载，等待一段时间后重试
                setTimeout(checkMarked, 100);
            }
        };
        checkMarked();
    }

    // 安全地渲染 Markdown
    renderMarkdown(content) {
        if (!content) return '';
        
        // 检查 marked 是否可用
        if (typeof marked === 'undefined') {
            console.warn('marked 库未加载，使用纯文本显示');
            return this.escapeHtml(content);
        }
        
        try {
            const html = marked.parse(content);
            return html;
        } catch (e) {
            console.error('Markdown 渲染失败:', e);
            return this.escapeHtml(content);
        }
    }

    // 高亮代码块
    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            try {
                container.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('hljs')) {
                        hljs.highlightElement(block);
                    }
                });
            } catch (e) {
                console.error('代码高亮失败:', e);
            }
        }
    }

    // 初始化DOM元素
    initializeElements() {
        // 侧边栏元素
        this.sidebar = document.querySelector('.sidebar');
        this.appLayout = document.getElementById('appLayout');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
        this.sidebarScrim = document.getElementById('sidebarScrim');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.aiOpsSidebarBtn = document.getElementById('aiOpsSidebarBtn');
        this.composerShell = document.getElementById('composerShell');
        this.promptRail = document.getElementById('promptRail');
        
        // 输入区域元素
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.toolsBtn = document.getElementById('toolsBtn');
        this.toolsMenu = document.getElementById('toolsMenu');
        this.uploadFileItem = document.getElementById('uploadFileItem');
        this.modeSelectorBtn = document.getElementById('modeSelectorBtn');
        this.modeDropdown = document.getElementById('modeDropdown');
        this.currentModeText = document.getElementById('currentModeText');
        this.fileInput = document.getElementById('fileInput');
        
        // 聊天区域元素
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.chatContainer = document.querySelector('.chat-container');
        this.welcomeGreeting = document.getElementById('welcomeGreeting');
        this.chatHistoryList = document.getElementById('chatHistoryList');
        this.contextMeter = document.getElementById('contextMeter');
        this.contextMeterFill = document.getElementById('contextMeterFill');
        this.contextMeterValue = document.getElementById('contextMeterValue');
        this.contextMeterTag = document.getElementById('contextMeterTag');

        // 知识点沉淀弹窗
        this.knowledgeModal = document.getElementById('knowledgeModal');
        this.knowledgeModalBackdrop = document.getElementById('knowledgeModalBackdrop');
        this.knowledgeModalClose = document.getElementById('knowledgeModalClose');
        this.knowledgeTitleInput = document.getElementById('knowledgeTitleInput');
        this.knowledgeContentInput = document.getElementById('knowledgeContentInput');
        this.knowledgeCancelBtn = document.getElementById('knowledgeCancelBtn');
        this.knowledgeConfirmBtn = document.getElementById('knowledgeConfirmBtn');
        this._precipitateTriggerBtn = null;
        // 草稿缓存 / 在途请求去重（按问答内容）
        this._knowledgeDraftCache = new Map();
        this._knowledgeDraftPending = new Map();
        this._knowledgeDraftKey = null;
        this._knowledgeDraftLoading = false;
        
        // 初始化时检查是否需要居中
        this.checkAndSetCentered();
    }

    // 绑定事件监听器
    bindEvents() {
        // 新建对话
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newChat());
        }
        
        // AI Ops按钮
        if (this.aiOpsSidebarBtn) {
            this.aiOpsSidebarBtn.addEventListener('click', () => this.triggerAIOps());
        }
        
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            item.addEventListener('click', () => {
                const mode = item.getAttribute('data-mode');
                this.selectMode(mode);
            });
        });
        
        // 发送消息
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => this.sendMessage());
        }
        
        if (this.messageInput) {
            this.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            this.messageInput.addEventListener('input', () => {
                this.autosizeComposer();
                this.syncSendAfford();
                this.refreshContextMeterFromLocal();
            });
        }

        if (this.composerShell) {
            this.composerShell.addEventListener('pointermove', (e) => {
                const rect = this.composerShell.getBoundingClientRect();
                this.composerShell.style.setProperty('--spot-x', `${e.clientX - rect.left}px`);
                this.composerShell.style.setProperty('--spot-y', `${e.clientY - rect.top}px`);
            });
            this.composerShell.addEventListener('pointerenter', () => {
                this.composerShell.classList.add('is-hot');
            });
            this.composerShell.addEventListener('pointerleave', () => {
                this.composerShell.classList.remove('is-hot');
            });
        }

        if (this.promptRail) {
            this.promptRail.querySelectorAll('.prompt-chip').forEach((chip) => {
                chip.addEventListener('click', () => {
                    const prompt = chip.getAttribute('data-prompt') || chip.textContent;
                    if (this.messageInput) {
                        this.messageInput.value = prompt;
                        this.autosizeComposer();
                        this.syncSendAfford();
                        this.messageInput.focus();
                    }
                    this.sendMessage();
                });
            });
        }

        if (this.sidebarToggle) {
            this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }
        if (this.sidebarCloseBtn) {
            this.sidebarCloseBtn.addEventListener('click', () => this.closeSidebar());
        }
        if (this.sidebarScrim) {
            this.sidebarScrim.addEventListener('click', () => this.closeSidebar());
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeToolsMenu();
                if (window.innerWidth <= 768) {
                    this.closeSidebar();
                }
                if (this.knowledgeModal && this.knowledgeModal.style.display !== 'none') {
                    if (!this._knowledgeDraftLoading) {
                        this.closeKnowledgeModal();
                    }
                }
            }
        });

        // 知识点沉淀弹窗
        if (this.knowledgeModalClose) {
            this.knowledgeModalClose.addEventListener('click', () => this.closeKnowledgeModal());
        }
        if (this.knowledgeCancelBtn) {
            this.knowledgeCancelBtn.addEventListener('click', () => this.closeKnowledgeModal());
        }
        if (this.knowledgeModalBackdrop) {
            this.knowledgeModalBackdrop.addEventListener('click', () => {
                // 生成中禁止点遮罩关闭，避免误关后重复请求
                if (this._knowledgeDraftLoading) {
                    this.showNotification('草稿生成中，请稍候…', 'warning');
                    return;
                }
                this.closeKnowledgeModal();
            });
        }
        if (this.knowledgeConfirmBtn) {
            this.knowledgeConfirmBtn.addEventListener('click', () => this.confirmKnowledgePrecipitate());
        }
        
        // 工具按钮和菜单
        if (this.toolsBtn) {
            this.toolsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleToolsMenu();
            });
        }
        
        // 工具菜单项点击事件
        if (this.uploadFileItem) {
            this.uploadFileItem.addEventListener('click', () => {
                if (this.fileInput) {
                    this.fileInput.click();
                }
                this.closeToolsMenu();
            });
        }
        
        // 点击外部关闭工具菜单
        document.addEventListener('click', (e) => {
            if (this.toolsBtn && this.toolsMenu && 
                !this.toolsBtn.contains(e.target) && 
                !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });
        
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    // 切换工具菜单显示/隐藏
    toggleToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭工具菜单
    closeToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 新建对话
    newChat() {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再新建对话', 'warning');
            return;
        }
        
        // 如果当前有对话内容，且不是从历史记录加载的，才保存为新的历史对话
        // 如果是从历史记录加载的，只需要更新该历史记录
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) {
                // 当前对话是从历史记录加载的，更新该历史记录
                this.updateCurrentChatHistory();
            } else {
                // 当前对话是新对话，保存为新的历史对话
                this.saveCurrentChat();
            }
        }
        
        // 停止所有进行中的操作
        this.isStreaming = false;
        
        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
            this.autosizeComposer();
        }
        
        // 清空当前对话历史
        this.currentChatHistory = [];
        
        // 重置标记
        this.isCurrentChatFromHistory = false;
        this._isAIOpsSession = false;
        this._aiopsHistorySaved = false;
        
        // 清空聊天记录
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
        }
        
        // 生成新的会话ID
        this.sessionId = this.generateSessionId();
        this._contextBaseTokens = 0;
        this.renderContextMeter(0, { instant: true });
        
        // 重置模式为快速
        this.currentMode = 'quick';
        this.updateUI();
        
        // 重新设置居中样式（确保对话框居中显示）
        this.checkAndSetCentered();
        
        // 确保容器有过渡动画
        if (this.chatContainer) {
            this.chatContainer.style.transition = 'all 0.5s ease';
        }
        
        // 更新历史对话列表
        this.renderChatHistory();
    }
    
    // 保存当前对话到历史记录（新建）
    saveCurrentChat() {
        if (this.currentChatHistory.length === 0) {
            return;
        }
        
        // 检查是否已存在相同ID的历史记录
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex !== -1) {
            // 如果已存在，更新而不是新建
            this.updateCurrentChatHistory();
            return;
        }
        
        // 获取对话标题（使用第一条用户消息的前30个字符）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        let title;
        if (firstUserMessage) {
            title = firstUserMessage.content.substring(0, 30)
                + (firstUserMessage.content.length > 30 ? '...' : '');
        } else if (this._isAIOpsSession) {
            title = 'AIOps 智能运维诊断';
        } else {
            title = '新对话';
        }
        
        const chatHistory = {
            id: this.sessionId,
            title: title,
            messages: [...this.currentChatHistory],
            type: this._isAIOpsSession ? 'aiops' : 'chat',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        // 添加到历史记录列表的开头
        this.chatHistories.unshift(chatHistory);
        
        // 限制历史记录数量（最多保存50条）
        if (this.chatHistories.length > 50) {
            this.chatHistories = this.chatHistories.slice(0, 50);
        }
        
        // 保存到localStorage
        this.saveChatHistories();
    }
    
    // 更新当前对话的历史记录
    updateCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) {
            return;
        }
        
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) {
            // 如果不存在，调用保存方法
            this.saveCurrentChat();
            return;
        }
        
        // 更新现有的历史记录
        const history = this.chatHistories[existingIndex];
        history.messages = [...this.currentChatHistory];
        history.updatedAt = new Date().toISOString();
        if (this._isAIOpsSession) {
            history.type = 'aiops';
        }
        
        // 如果标题需要更新（第一条消息改变了）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            const newTitle = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
            if (history.title !== newTitle) {
                history.title = newTitle;
            }
        }
        
        // 保存到localStorage
        this.saveChatHistories();
    }
    
    // 加载历史对话列表
    loadChatHistories() {
        try {
            const stored = localStorage.getItem('chatHistories');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('加载历史对话失败:', e);
            return [];
        }
    }
    
    // 保存历史对话列表到localStorage
    saveChatHistories() {
        try {
            localStorage.setItem('chatHistories', JSON.stringify(this.chatHistories));
        } catch (e) {
            console.error('保存历史对话失败:', e);
            // 报告过大时降级：只保留最近摘要，避免整表丢失
            try {
                const compact = this.chatHistories.map(h => {
                    if (h.type !== 'aiops') return h;
                    return {
                        ...h,
                        messages: (h.messages || []).map(m => ({
                            ...m,
                            content: typeof m.content === 'string' && m.content.length > 20000
                                ? (m.content.slice(0, 20000) + '\n\n…(内容过长已截断)')
                                : m.content
                        }))
                    };
                });
                localStorage.setItem('chatHistories', JSON.stringify(compact));
                this.chatHistories = compact;
            } catch (e2) {
                console.error('压缩后仍无法保存历史:', e2);
                if (typeof this.showNotification === 'function') {
                    this.showNotification('历史记录过大，无法写入本地存储', 'warning');
                }
            }
        }
    }
    
    // 渲染历史对话列表
    renderChatHistory() {
        if (!this.chatHistoryList) {
            return;
        }
        
        this.chatHistoryList.innerHTML = '';
        
        if (this.chatHistories.length === 0) {
            return;
        }
        
        this.chatHistories.forEach((history, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.historyId = history.id;
            
            if (history.id === this.sessionId) {
                historyItem.classList.add('active');
            }
            historyItem.dataset.type = history.type || 'chat';
            historyItem.innerHTML = `
                <div class="history-item-content">
                    <span class="history-type-dot" aria-hidden="true"></span>
                    <span class="history-item-title">${this.escapeHtml(history.title)}</span>
                </div>
                <button class="history-item-delete" data-history-id="${history.id}" title="删除">
                    <i class="ph ph-x"></i>
                </button>
            `;
            
            // 点击历史项加载对话
            historyItem.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-delete')) {
                    this.loadChatHistory(history.id);
                }
            });
            
            // 删除历史对话
            const deleteBtn = historyItem.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChatHistory(history.id);
            });
            
            this.chatHistoryList.appendChild(historyItem);
        });
    }
    
    // 加载历史对话
    async loadChatHistory(historyId) {
        const history = this.chatHistories.find(h => h.id === historyId);
        if (!history) {
            return;
        }
        
        // 如果当前有对话内容，且不是同一个对话，先保存
        if (this.currentChatHistory.length > 0 && this.sessionId !== historyId) {
            if (this.isCurrentChatFromHistory) {
                this.updateCurrentChatHistory();
            } else {
                this.saveCurrentChat();
            }
        }

        const renderLocalHistory = () => {
            this.sessionId = history.id;
            this.isCurrentChatFromHistory = true;
            this._isAIOpsSession = history.type === 'aiops';
            this.currentChatHistory = [...(history.messages || [])];
            if (this.chatMessages) {
                this.chatMessages.innerHTML = '';
                this.currentChatHistory.forEach(msg => {
                    const type = msg.type === 'user' ? 'user' : 'assistant';
                    if (this._isAIOpsSession && type === 'assistant') {
                        this.addAIOpsMessage(msg.content, []);
                    } else {
                        this.addMessage(type, msg.content, false, false);
                    }
                });
            }
        };

        // AIOps 仅保存在本地，不走 RAG 会话接口
        if (history.type === 'aiops') {
            renderLocalHistory();
            this._contextBaseTokens = this._estimateHistoryTokens();
            this.refreshContextMeterFromLocal();
            this.checkAndSetCentered();
            this.renderChatHistory();
            this.updateUI();
            return;
        }
        
        try {
            const response = await fetch(`/api/chat/session/${historyId}`);
            if (response.ok) {
                const data = await response.json();
                const backendHistory = data.history || [];
                
                this.sessionId = history.id;
                this.isCurrentChatFromHistory = true;
                this._isAIOpsSession = false;
                
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';
                    
                    if (backendHistory.length > 0) {
                        this.currentChatHistory = [];
                        backendHistory.forEach(msg => {
                            const messageType = msg.role === 'user' ? 'user' : 'assistant';
                            this.addMessage(messageType, msg.content, false, false);
                            this.currentChatHistory.push({
                                type: messageType,
                                content: msg.content,
                                timestamp: msg.timestamp || new Date().toISOString()
                            });
                        });
                    } else {
                        this.currentChatHistory = [...history.messages];
                        history.messages.forEach(msg => {
                            this.addMessage(msg.type, msg.content, false, false);
                        });
                    }
                }
            } else {
                console.warn('从后端加载历史失败，使用本地缓存');
                renderLocalHistory();
            }
        } catch (error) {
            console.error('加载会话历史失败:', error);
            renderLocalHistory();
        }
        
        this.checkAndSetCentered();
        this.renderChatHistory();
        this.updateUI();
        this._contextBaseTokens = this._estimateHistoryTokens();
        this.refreshContextMeterFromLocal();
    }
    
    // 删除历史对话
    async deleteChatHistory(historyId) {
        try {
            // 调用后端API清空会话
            const response = await fetch('/api/chat/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: historyId
                })
            });

            if (!response.ok) {
                throw new Error('清空会话失败');
            }

            const result = await response.json();
            
            if (result.status === 'success') {
                // 从本地存储中删除
                this.chatHistories = this.chatHistories.filter(h => h.id !== historyId);
                this.saveChatHistories();
                this.renderChatHistory();
                
                // 如果删除的是当前对话，清空当前对话
                if (this.sessionId === historyId) {
                    this.currentChatHistory = [];
                    if (this.chatMessages) {
                        this.chatMessages.innerHTML = '';
                    }
                    this.sessionId = this.generateSessionId();
                    this._contextBaseTokens = 0;
                    this.renderContextMeter(0, { instant: true });
                    this.checkAndSetCentered();
                }
                
                this.showNotification('会话已清空', 'success');
            } else {
                throw new Error(result.message || '清空会话失败');
            }
        } catch (error) {
            console.error('删除历史对话失败:', error);
            this.showNotification('删除失败: ' + error.message, 'error');
        }
    }

    // 切换模式下拉菜单
    toggleModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭模式下拉菜单
    closeModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 选择模式
    selectMode(mode) {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再切换模式', 'warning');
            return;
        }
        
        this.currentMode = mode;
        this.updateUI();
        
        const modeNames = {
            'quick': '快速',
            'stream': '流式'
        };
        
        this.showNotification(`已切换到${modeNames[mode]}模式`, 'info');
    }

    // 更新UI
    updateUI() {
        // 更新模式选择器显示
        if (this.currentModeText) {
            const modeNames = {
                'quick': '快速',
                'stream': '流式'
            };
            this.currentModeText.textContent = modeNames[this.currentMode] || '快速';
        }
        
        // 更新下拉菜单选中状态
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            const mode = item.getAttribute('data-mode');
            if (mode === this.currentMode) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // 更新发送按钮状态
        if (this.sendButton) {
            this.sendButton.disabled = this.isStreaming;
        }

        if (this.messageInput) {
            this.messageInput.disabled = this.isStreaming;
            this.messageInput.placeholder = '描述现象、贴日志，或直接提问';
        }
        this.syncSendAfford();
    }

    // 生成随机会话ID
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // 发送消息
    async sendMessage() {
        let message = '';
        if (this.messageInput) {
            message = this.messageInput.value.trim();
        }
        
        if (!message) {
            this.showNotification('请输入消息内容', 'warning');
            return;
        }

        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成', 'warning');
            return;
        }

        // 显示用户消息
        this.addMessage('user', message);
        
        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
            this.autosizeComposer();
        }

        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            if (this.currentMode === 'quick') {
                await this.sendQuickMessage(message);
            } else if (this.currentMode === 'stream') {
                await this.sendStreamMessage(message);
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            this.addMessage('assistant', '抱歉，发送消息时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();
            
            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory && this.currentChatHistory.length > 0) {
                this.updateCurrentChatHistory();
                this.renderChatHistory(); // 更新历史对话列表显示
            }
        }
    }

    // 发送快速消息（普通对话）
    async sendQuickMessage(message) {
        // 添加等待提示消息
        const loadingMessage = this.addLoadingMessage('正在思考...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            console.log('[sendQuickMessage] 响应数据:', JSON.stringify(data));
            
            // 移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }
            
            // 统一响应格式：检查 data.code 或 data.message 判断请求是否成功
            if (data.code === 200 || data.message === 'success') {
                // data.data 是 ChatResponse 对象
                const chatResponse = data.data;
                
                if (chatResponse && chatResponse.success) {
                    // 成功：添加实际响应消息（即使 answer 为空也显示）
                    const answer = chatResponse.answer || '（无回复内容）';
                    const assistantEl = this.addMessage('assistant', answer);
                    this.attachRagSources(assistantEl, chatResponse.rag);
                    this.attachPrecipitateButton(assistantEl, message, answer);
                    this.applyContextSnapshot(chatResponse.context);
                } else if (chatResponse && chatResponse.errorMessage) {
                    // 业务错误
                    throw new Error(chatResponse.errorMessage);
                } else {
                    // 兜底：尝试显示任何可用内容
                    const fallbackAnswer = chatResponse?.answer || chatResponse?.errorMessage || '服务返回了空内容';
                    this.addMessage('assistant', fallbackAnswer);
                }
            } else {
                // HTTP 成功但业务失败
                throw new Error(data.message || '请求失败');
            }
        } catch (error) {
            // 出错时也要移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }
            throw error;
        }
    }

    // 发送流式消息
    async sendStreamMessage(message) {
        // 先挂载助手气泡（含头像），避免等数据才出现
        const assistantMessageElement = this.addLoadingMessage('思考中...');
        let fullResponse = '';
        let ragMeta = null;

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat_stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            // 处理流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        // 流结束，使用统一的处理方法
                        this.handleStreamComplete(assistantMessageElement, fullResponse, ragMeta);
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });
                    
                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.trim() === '') continue;
                        
                        console.log('[SSE调试] 收到行:', line);
                        
                        // 解析SSE格式
                        if (line.startsWith('id:')) {
                            console.log('[SSE调试] 解析到ID');
                            continue;
                        } else if (line.startsWith('event:')) {
                            // 兼容 "event:message" 和 "event: message" 两种格式
                            currentEvent = line.substring(6).trim();
                            console.log('[SSE调试] 解析到事件类型:', currentEvent);
                            // 注意：后端统一使用 "message" 事件名，真正的类型在 data 的 JSON 中
                            continue;
                        } else if (line.startsWith('data:')) {
                            // 兼容 "data:xxx" 和 "data: xxx" 两种格式
                            const rawData = line.substring(5).trim();
                            console.log('[SSE调试] 解析到数据, currentEvent:', currentEvent, ', rawData:', rawData);
                            
                            // 兼容旧格式 [DONE] 标记
                            if (rawData === '[DONE]') {
                                // 流结束标记，将内容转换为Markdown渲染
                                this.handleStreamComplete(assistantMessageElement, fullResponse, ragMeta);
                                return;
                            }
                            
                            // 处理 SSE 数据
                            try {
                                // 尝试解析为 SseMessage 格式的 JSON
                                const sseMessage = JSON.parse(rawData);
                                console.log('[SSE调试] 解析JSON成功:', sseMessage);
                                
                                if (sseMessage && typeof sseMessage.type === 'string') {
                                    if (sseMessage.type === 'status') {
                                        const statusData = sseMessage.data || {};
                                        this.updateAssistantLoadingText(
                                            assistantMessageElement,
                                            statusData.message || '思考中…'
                                        );
                                    } else if (sseMessage.type === 'context') {
                                        this.applyContextSnapshot(sseMessage.data);
                                    } else if (sseMessage.type === 'tool_call') {
                                        const toolData = sseMessage.data || {};
                                        if (toolData.tool === 'retrieve_knowledge') {
                                            if (toolData.status === 'start') {
                                                this.updateAssistantLoadingText(
                                                    assistantMessageElement,
                                                    toolData.message || '正在检索知识库…'
                                                );
                                            } else if (toolData.status === 'end') {
                                                this.setRagStatus(
                                                    assistantMessageElement,
                                                    toolData.message
                                                        || (toolData.hit
                                                            ? `已从知识库命中 ${toolData.count || 0} 条`
                                                            : '知识库未命中相关内容'),
                                                    toolData.hit ? 'hit' : 'miss'
                                                );
                                            }
                                        }
                                    } else if (sseMessage.type === 'search_results') {
                                        ragMeta = sseMessage.data || null;
                                        if (ragMeta) {
                                            this.setRagStatus(
                                                assistantMessageElement,
                                                ragMeta.message
                                                    || (ragMeta.hit
                                                        ? `已从知识库命中 ${ragMeta.count || 0} 条`
                                                        : '知识库未命中相关内容'),
                                                ragMeta.hit ? 'hit' : 'miss'
                                            );
                                            if (ragMeta.hit) {
                                                this.attachRagSources(assistantMessageElement, ragMeta);
                                            }
                                        }
                                    } else if (sseMessage.type === 'content') {
                                        const content = sseMessage.data || '';
                                        fullResponse += content;
                                        console.log('[SSE调试] 添加内容:', content);
                                        
                                        // 实时渲染 Markdown
                                        if (assistantMessageElement) {
                                            this.finishAssistantLoading(assistantMessageElement);
                                            assistantMessageElement.classList.add('streaming');
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            if (messageContent) {
                                                messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                                this.highlightCodeBlocks(messageContent);
                                            }
                                            this.scrollToBottom();
                                        }
                                    } else if (sseMessage.type === 'done' || sseMessage.type === 'complete') {
                                        console.log('[SSE调试] 收到完成标记，流结束');
                                        if (sseMessage.data && sseMessage.data.rag) {
                                            ragMeta = sseMessage.data.rag;
                                        }
                                        if (sseMessage.data && sseMessage.data.context) {
                                            this.applyContextSnapshot(sseMessage.data.context);
                                        }
                                        this.handleStreamComplete(assistantMessageElement, fullResponse, ragMeta);
                                        return;
                                    } else if (sseMessage.type === 'error') {
                                        console.error('[SSE调试] 收到错误:', sseMessage.data);
                                        if (assistantMessageElement) {
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            messageContent.innerHTML = this.renderMarkdown('错误: ' + (sseMessage.data || '未知错误'));
                                        }
                                        return;
                                    }
                                } else {
                                    // 不是标准 SseMessage 格式，尝试兼容处理
                                    console.log('[SSE调试] 非标准格式，尝试兼容处理');
                                    fullResponse += rawData;
                                    if (assistantMessageElement) {
                                        const messageContent = assistantMessageElement.querySelector('.message-content');
                                        messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                        this.highlightCodeBlocks(messageContent);
                                        this.scrollToBottom();
                                    }
                                }
                            } catch (e) {
                                // JSON 解析失败，尝试兼容旧格式
                                console.log('[SSE调试] JSON解析失败，使用兼容模式:', e.message);
                                if (rawData === '') {
                                    fullResponse += '\n';
                                } else {
                                    fullResponse += rawData;
                                }
                                
                                if (assistantMessageElement) {
                                    const messageContent = assistantMessageElement.querySelector('.message-content');
                                    messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                    this.highlightCodeBlocks(messageContent);
                                    this.scrollToBottom();
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            if (assistantMessageElement) {
                const messageContent = assistantMessageElement.querySelector('.message-content');
                if (messageContent) {
                    messageContent.classList.remove('loading-message-content');
                    messageContent.textContent = '抱歉，发送消息时出现错误：' + error.message;
                }
            }
            throw error;
        }
    }

    // 添加消息到聊天界面
    addMessage(type, content, isStreaming = false, saveToHistory = true) {
        // 检查是否是第一条消息，如果是则移除居中样式
        const isFirstMessage = this.chatMessages && this.chatMessages.querySelectorAll('.message').length === 0;
        
        // 保存消息到当前对话历史（如果不是流式消息且需要保存）
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({
                type: type,
                content: content,
                timestamp: new Date().toISOString()
            });
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}${isStreaming ? ' streaming' : ''}`;

        // 如果是assistant消息，添加头像图标
        if (type === 'assistant') {
            const messageAvatar = document.createElement('div');
            messageAvatar.className = 'message-avatar';
            messageAvatar.innerHTML = this.createAgentMark();
            messageDiv.appendChild(messageAvatar);
        }

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // 如果是assistant消息且不是流式消息，使用Markdown渲染
        if (type === 'assistant' && !isStreaming) {
            messageContent.innerHTML = this.renderMarkdown(content);
            // 高亮代码块
            this.highlightCodeBlocks(messageContent);
        } else {
            // 用户消息或流式消息使用纯文本
            messageContent.textContent = content;
        }

        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            
            // 如果是第一条消息，移除居中样式并添加动画
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                // 添加动画类
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            
            this.scrollToBottom();
        }

        return messageDiv;
    }

    // 添加带加载动画的消息
    addLoadingMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = this.createAgentMark();
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content loading-message-content';
        
        // 创建文本和动画容器
        const textSpan = document.createElement('span');
        textSpan.className = 'loading-text-span';
        textSpan.textContent = content;
        
        const loadingIcon = document.createElement('span');
        loadingIcon.className = 'loading-spinner-icon think-bars';
        loadingIcon.innerHTML = '<i></i><i></i><i></i>';
        
        messageContent.appendChild(textSpan);
        messageContent.appendChild(loadingIcon);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);
        messageDiv.classList.add('is-generating');
        messageAvatar.classList.add('avatar-generating');

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            
            // 如果是第一条消息，移除居中样式
            const isFirstMessage = this.chatMessages.querySelectorAll('.message').length === 1;
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            
            this.scrollToBottom();
        }

        return messageDiv;
    }
    
    // 检查并设置居中样式
    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            const hasMessages = this.chatMessages.querySelectorAll('.message').length > 0;
            if (!hasMessages) {
                this.chatContainer.classList.add('centered');
            } else {
                this.chatContainer.classList.remove('centered');
            }
        }
    }

    // 滚动到底部
    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }

    // 结束助手消息的加载态（去掉转圈/流式光标）
    finishAssistantLoading(messageElement) {
        if (!messageElement) {
            return;
        }
        messageElement.classList.remove('streaming', 'is-generating');
        const avatar = messageElement.querySelector('.message-avatar');
        if (avatar) {
            avatar.classList.remove('avatar-generating');
        }
        const messageContent = messageElement.querySelector('.message-content');
        if (messageContent) {
            messageContent.classList.remove('loading-message-content');
            const spinner = messageContent.querySelector('.loading-spinner-icon');
            if (spinner) {
                spinner.remove();
            }
        }
    }

    // 更新加载气泡文案（思考中 / 检索中）
    updateAssistantLoadingText(messageElement, text) {
        if (!messageElement || !text) {
            return;
        }
        const messageContent = messageElement.querySelector('.message-content');
        if (!messageContent) {
            return;
        }
        let textSpan = messageContent.querySelector('.loading-text-span');
        if (!textSpan) {
            // 仍处于 loading 结构时取第一个 span
            textSpan = messageContent.querySelector('span:not(.loading-spinner-icon)');
        }
        if (textSpan && messageContent.classList.contains('loading-message-content')) {
            textSpan.textContent = text;
        }
        this.setRagStatus(messageElement, text, 'searching');
    }

    // 处理流式传输完成
    handleStreamComplete(assistantMessageElement, fullResponse, ragMeta = null) {
        if (assistantMessageElement) {
            this.finishAssistantLoading(assistantMessageElement);
            const messageContent = assistantMessageElement.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = this.renderMarkdown(fullResponse || '');
                // 高亮代码块
                this.highlightCodeBlocks(messageContent);
            }
            if (ragMeta && ragMeta.hit) {
                this.setRagStatus(
                    assistantMessageElement,
                    ragMeta.message || `已从知识库命中 ${ragMeta.count || 0} 条`,
                    'hit'
                );
                this.attachRagSources(assistantMessageElement, ragMeta);
            } else if (ragMeta && ragMeta.message) {
                this.setRagStatus(assistantMessageElement, ragMeta.message, 'miss');
            }
        }
        // 保存流式消息到历史记录
        if (fullResponse) {
            this.currentChatHistory.push({
                type: 'assistant',
                content: fullResponse,
                timestamp: new Date().toISOString(),
                rag: ragMeta || null
            });
            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory) {
                this.updateCurrentChatHistory();
                this.renderChatHistory();
            }

            const question = this.getLastUserQuestion();
            if (question && assistantMessageElement) {
                this.attachPrecipitateButton(assistantMessageElement, question, fullResponse);
            }
        }
    }

    // RAG 过程提示（检索中 / 命中 / 未命中）
    setRagStatus(messageElement, text, state) {
        if (!messageElement || !text) {
            return;
        }
        const wrapper = messageElement.querySelector('.message-content-wrapper');
        if (!wrapper) {
            return;
        }
        let statusEl = wrapper.querySelector('.rag-status');
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.className = 'rag-status';
            const contentEl = wrapper.querySelector('.message-content');
            const badgeWrap = wrapper.querySelector('.rag-badge-wrap');
            // 状态始终放在最上方，方便先看到「正在检索」
            if (badgeWrap) {
                wrapper.insertBefore(statusEl, badgeWrap);
            } else if (contentEl) {
                wrapper.insertBefore(statusEl, contentEl);
            } else {
                wrapper.appendChild(statusEl);
            }
        }
        statusEl.className = `rag-status rag-status-${state || 'info'}`;
        statusEl.textContent = text;
        // 检索中时确保气泡可见（内容还可能为空）
        if (state === 'searching') {
            const contentEl = wrapper.querySelector('.message-content');
            if (contentEl && !contentEl.textContent.trim() && !contentEl.innerHTML.trim()) {
                contentEl.innerHTML = '<span class="rag-waiting-placeholder"> </span>';
            }
        }
        this.scrollToBottom();
    }

    // 命中知识库时展示可展开来源标签
    attachRagSources(messageElement, ragMeta) {
        if (!messageElement || !ragMeta || !ragMeta.hit) {
            return;
        }
        const wrapper = messageElement.querySelector('.message-content-wrapper');
        if (!wrapper) {
            return;
        }

        const count = ragMeta.count || (ragMeta.sources || []).length || 0;
        const sources = ragMeta.sources || [];

        let badgeWrap = wrapper.querySelector('.rag-badge-wrap');
        if (!badgeWrap) {
            badgeWrap = document.createElement('div');
            badgeWrap.className = 'rag-badge-wrap';
            const contentEl = wrapper.querySelector('.message-content');
            const statusEl = wrapper.querySelector('.rag-status');
            if (contentEl) {
                wrapper.insertBefore(badgeWrap, contentEl);
            } else if (statusEl && statusEl.nextSibling) {
                wrapper.insertBefore(badgeWrap, statusEl.nextSibling);
            } else {
                wrapper.appendChild(badgeWrap);
            }
        }

        badgeWrap.innerHTML = '';

        const badge = document.createElement('button');
        badge.type = 'button';
        badge.className = 'rag-badge';
        badge.textContent = `来自知识库 · ${count} 条`;

        const list = document.createElement('div');
        list.className = 'rag-sources';
        list.style.display = 'none';

        if (sources.length === 0) {
            list.innerHTML = '<div class="rag-source-item">暂无来源详情</div>';
        } else {
            sources.forEach((src, idx) => {
                const item = document.createElement('div');
                item.className = 'rag-source-item';
                const title = src.title ? this.escapeHtml(src.title) : '';
                const source = this.escapeHtml(src.source || '未知来源');
                item.innerHTML = `
                    <div class="rag-source-index">#${idx + 1}</div>
                    <div class="rag-source-main">
                        <div class="rag-source-file">${source}</div>
                        ${title ? `<div class="rag-source-title">${title}</div>` : ''}
                    </div>
                `;
                list.appendChild(item);
            });
        }

        badge.addEventListener('click', () => {
            const open = list.style.display !== 'none';
            list.style.display = open ? 'none' : 'block';
            badge.classList.toggle('expanded', !open);
        });

        badgeWrap.appendChild(badge);
        badgeWrap.appendChild(list);
    }

    // 获取最近一条用户问题
    getLastUserQuestion() {
        for (let i = this.currentChatHistory.length - 1; i >= 0; i--) {
            const item = this.currentChatHistory[i];
            if (item && item.type === 'user' && item.content) {
                return item.content;
            }
        }
        return '';
    }

    // 在助手消息下挂载「沉淀为知识点」按钮
    attachPrecipitateButton(messageElement, question, answer) {
        if (!messageElement || !question || !answer) {
            return;
        }
        if (messageElement.querySelector('.precipitate-btn')) {
            return;
        }

        const wrapper = messageElement.querySelector('.message-content-wrapper') || messageElement;
        const actions = document.createElement('div');
        actions.className = 'message-actions';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'precipitate-btn';
        btn.textContent = '沉淀为知识点';
        btn.addEventListener('click', () => {
            this.openKnowledgeDraft(question, answer, btn);
        });

        actions.appendChild(btn);
        wrapper.appendChild(actions);
    }

    _knowledgeDraftCacheKey(question, answer) {
        return `${question}\n---\n${answer}`;
    }

    _fillKnowledgeDraftForm(title, content, loading) {
        if (this.knowledgeTitleInput) {
            this.knowledgeTitleInput.value = title || '';
            this.knowledgeTitleInput.disabled = loading;
        }
        if (this.knowledgeContentInput) {
            this.knowledgeContentInput.value = content || '';
            this.knowledgeContentInput.disabled = loading;
        }
        if (this.knowledgeConfirmBtn) {
            this.knowledgeConfirmBtn.disabled = loading;
        }
        this._knowledgeDraftLoading = loading;
    }

    async _requestKnowledgeDraft(question, answer) {
        const response = await fetch(`${this.apiBaseUrl}/knowledge/draft`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: this.sessionId,
                question: question,
                answer: answer
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }

        const data = await response.json();
        if (data.code !== 200 || !data.data || !data.data.content) {
            const errMsg = (data.data && data.data.errorMessage) || data.message || '生成草稿失败';
            throw new Error(errMsg);
        }

        return {
            title: data.data.title || '',
            content: data.data.content || ''
        };
    }

    // 打开沉淀弹窗并请求 LLM 草稿（同问答复用缓存 / 在途请求）
    async openKnowledgeDraft(question, answer, triggerBtn) {
        if (!this.knowledgeModal) {
            return;
        }

        this._precipitateTriggerBtn = triggerBtn || null;
        const key = this._knowledgeDraftCacheKey(question, answer);
        this._knowledgeDraftKey = key;
        this.knowledgeModal.style.display = 'flex';

        // 已有缓存：直接回填，不重复请求
        const cached = this._knowledgeDraftCache.get(key);
        if (cached) {
            this._fillKnowledgeDraftForm(cached.title, cached.content, false);
            return;
        }

        this._fillKnowledgeDraftForm('', '正在生成知识点草稿，请稍候...', true);

        // 复用在途请求，避免重复打接口
        let pending = this._knowledgeDraftPending.get(key);
        if (!pending) {
            pending = this._requestKnowledgeDraft(question, answer)
                .then((draft) => {
                    this._knowledgeDraftCache.set(key, draft);
                    return draft;
                })
                .finally(() => {
                    this._knowledgeDraftPending.delete(key);
                });
            this._knowledgeDraftPending.set(key, pending);
        }

        try {
            const draft = await pending;
            // 仅当仍在查看同一条问答时回填（弹窗关闭后也已写入缓存）
            if (this._knowledgeDraftKey === key) {
                this._fillKnowledgeDraftForm(draft.title, draft.content, false);
            }
        } catch (error) {
            console.error('生成知识点草稿失败:', error);
            if (this._knowledgeDraftKey === key) {
                this._knowledgeDraftLoading = false;
                this.showNotification('生成草稿失败：' + error.message, 'error');
                this.closeKnowledgeModal();
            }
        }
    }

    closeKnowledgeModal() {
        // 关闭前保留已编辑内容，再次打开可回填
        if (
            this._knowledgeDraftKey &&
            this.knowledgeTitleInput &&
            !this.knowledgeTitleInput.disabled
        ) {
            this._knowledgeDraftCache.set(this._knowledgeDraftKey, {
                title: this.knowledgeTitleInput.value,
                content: this.knowledgeContentInput ? this.knowledgeContentInput.value : ''
            });
        }

        if (this.knowledgeModal) {
            this.knowledgeModal.style.display = 'none';
        }
        if (this.knowledgeConfirmBtn) {
            this.knowledgeConfirmBtn.disabled = false;
        }
        if (this.knowledgeTitleInput) {
            this.knowledgeTitleInput.disabled = false;
        }
        if (this.knowledgeContentInput) {
            this.knowledgeContentInput.disabled = false;
        }
        // 生成中关闭弹窗不取消请求；再次打开会走缓存/在途复用
        this._knowledgeDraftLoading = false;
    }

    // 确认入库
    async confirmKnowledgePrecipitate() {
        const title = this.knowledgeTitleInput ? this.knowledgeTitleInput.value.trim() : '';
        const content = this.knowledgeContentInput ? this.knowledgeContentInput.value.trim() : '';

        if (!title || !content) {
            this.showNotification('请填写标题和内容', 'warning');
            return;
        }

        if (this.knowledgeConfirmBtn) {
            this.knowledgeConfirmBtn.disabled = true;
            this.knowledgeConfirmBtn.textContent = '入库中...';
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/knowledge/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title,
                    content: content,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            if (data.code !== 200 || !data.data || !data.data.success) {
                const errMsg = (data.data && data.data.errorMessage) || data.message || '入库失败';
                throw new Error(errMsg);
            }

            const chunkCount = data.data.chunk_count || 0;
            this.showNotification(
                `知识点已入库（${data.data.filename}，${chunkCount} 个分片）`,
                'success'
            );

            if (this._precipitateTriggerBtn) {
                this._precipitateTriggerBtn.disabled = true;
                this._precipitateTriggerBtn.classList.add('done');
                this._precipitateTriggerBtn.textContent = '已沉淀';
            }

            if (this._knowledgeDraftKey) {
                this._knowledgeDraftCache.delete(this._knowledgeDraftKey);
                this._knowledgeDraftKey = null;
            }

            this.closeKnowledgeModal();
        } catch (error) {
            console.error('知识点入库失败:', error);
            this.showNotification('入库失败：' + error.message, 'error');
            if (this.knowledgeConfirmBtn) {
                this.knowledgeConfirmBtn.disabled = false;
            }
        } finally {
            if (this.knowledgeConfirmBtn) {
                this.knowledgeConfirmBtn.textContent = '确认入库';
            }
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'toastOut 0.28s ease forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 280);
        }, 3000);
    }

    initShellChrome() {
        const collapsed = localStorage.getItem('intelliops.rail') === 'min';
        if (collapsed && this.appLayout && window.innerWidth > 768) {
            this.appLayout.classList.add('rail-min');
        }
        this.syncSendAfford();
    }

    _estimateHistoryTokens() {
        return (this.currentChatHistory || []).reduce(
            (sum, msg) => sum + String(msg.content || '').length,
            0
        );
    }

    _formatTokenCount(count) {
        if (count >= 1000) {
            return `${(count / 1000).toFixed(1).replace(/\.0$/, '')}k`;
        }
        return String(count);
    }

    refreshContextMeterFromLocal() {
        const draft = (this.messageInput && this.messageInput.value) ? this.messageInput.value.length : 0;
        const tokens = this._contextBaseTokens > 0
            ? this._contextBaseTokens + draft
            : this._estimateHistoryTokens() + draft;
        this.renderContextMeter(tokens);
    }

    applyContextSnapshot(data) {
        if (!data || typeof data !== 'object') {
            this.refreshContextMeterFromLocal();
            return;
        }
        if (data.max_tokens) {
            this._contextMaxTokens = data.max_tokens;
        }
        const nextTokens = Number(data.tokens) || 0;
        const previous = Number(data.tokens_before);
        if (data.compressed && previous > nextTokens) {
            this.renderContextMeter(previous, { instant: true });
            requestAnimationFrame(() => {
                this.renderContextMeter(nextTokens, { compressed: true });
            });
        } else {
            this.renderContextMeter(nextTokens, { compressed: !!data.compressed });
        }
        this._contextBaseTokens = nextTokens;
    }

    renderContextMeter(tokens, options = {}) {
        const meter = this.contextMeter;
        const fill = this.contextMeterFill;
        const valueEl = this.contextMeterValue;
        if (!meter || !fill || !valueEl) {
            return;
        }
        const maxTokens = Math.max(this._contextMaxTokens || 24000, 1);
        const safeTokens = Math.max(0, Number(tokens) || 0);
        const percent = Math.min(100, Math.round((safeTokens / maxTokens) * 100));
        let level = 'low';
        if (percent >= 85) {
            level = 'high';
        } else if (percent >= 60) {
            level = 'mid';
        }
        if (options.instant) {
            fill.style.transition = 'none';
        } else {
            fill.style.transition = '';
        }
        fill.style.width = `${percent}%`;
        meter.dataset.level = level;
        valueEl.textContent = `${percent}%`;
        const label = `上下文 ${this._formatTokenCount(safeTokens)} / ${this._formatTokenCount(maxTokens)}`;
        meter.title = label;
        meter.setAttribute('aria-label', `${label}，占用 ${percent}%`);
        if (options.compressed && this.contextMeterTag) {
            meter.classList.remove('is-compressing');
            void meter.offsetWidth;
            meter.classList.add('is-compressing');
            this.contextMeterTag.hidden = false;
            this.contextMeterTag.classList.add('is-on');
            clearTimeout(this._contextTagTimer);
            this._contextTagTimer = setTimeout(() => {
                if (this.contextMeterTag) {
                    this.contextMeterTag.hidden = true;
                    this.contextMeterTag.classList.remove('is-on');
                }
                meter.classList.remove('is-compressing');
            }, 2400);
        }
    }

    toggleSidebar() {
        if (!this.appLayout) {
            return;
        }
        if (window.innerWidth <= 768) {
            const open = this.appLayout.classList.toggle('rail-open');
            if (this.sidebarScrim) {
                this.sidebarScrim.hidden = !open;
            }
            return;
        }
        const collapsed = this.appLayout.classList.toggle('rail-min');
        localStorage.setItem('intelliops.rail', collapsed ? 'min' : 'open');
    }

    closeSidebar() {
        if (!this.appLayout) {
            return;
        }
        if (window.innerWidth <= 768) {
            this.appLayout.classList.remove('rail-open');
            if (this.sidebarScrim) {
                this.sidebarScrim.hidden = true;
            }
            return;
        }
        this.appLayout.classList.add('rail-min');
        localStorage.setItem('intelliops.rail', 'min');
    }

    autosizeComposer() {
        if (!this.messageInput) {
            return;
        }
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = `${Math.min(this.messageInput.scrollHeight, 160)}px`;
    }

    syncSendAfford() {
        if (!this.sendButton) {
            return;
        }
        const hasText = !!(this.messageInput && this.messageInput.value.trim());
        this.sendButton.classList.toggle('has-text', hasText && !this.isStreaming);
    }

    // 处理文件选择
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // 验证文件格式
            if (!this.validateFileType(file)) {
                this.showNotification('只支持上传 TXT 或 Markdown (.md) 格式的文件', 'error');
                this.fileInput.value = '';
                return;
            }
            this.uploadFile(file);
        }
    }

    // 验证文件类型
    validateFileType(file) {
        const fileName = file.name.toLowerCase();
        const allowedExtensions = ['.txt', '.md', '.markdown'];
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    }

    // 上传文件到知识库
    async uploadFile(file) {
        // 再次验证文件类型（双重保险）
        if (!this.validateFileType(file)) {
            this.showNotification('只支持上传 TXT 或 Markdown (.md) 格式的文件', 'error');
            return;
        }

        // 验证文件大小（限制为50MB）
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('文件大小不能超过50MB', 'error');
            return;
        }

        // 锁定前端并显示上传遮罩层
        this.isStreaming = true;
        this.updateUI();
        this.showUploadOverlay(true, file.name);

        try {
            // 创建 FormData
            const formData = new FormData();
            formData.append('file', file);

            // 发送上传请求
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();

            if ((data.code === 200 || data.message === 'success') && data.data) {
                // 在聊天界面显示上传成功消息
                const successMessage = `${file.name} 上传到知识库成功`;
                this.addMessage('assistant', successMessage, false, true);
            } else {
                throw new Error(data.message || '上传失败');
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.showNotification('文件上传失败: ' + error.message, 'error');
        } finally {
            // 清空文件输入
            if (this.fileInput) {
                this.fileInput.value = '';
            }
            // 解锁前端
            this.isStreaming = false;
            this.showUploadOverlay(false);
            this.updateUI();
        }
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    // 发送智能运维请求（SSE 流式模式）
    async sendAIOpsRequest(loadingMessageElement) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/aiops`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            let fullResponse = '';

            // 处理 SSE 流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = 'message'; // 默认事件类型为 message

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        // 流结束，更新最终内容
                        if (fullResponse) {
                            console.log('AI Ops 流结束，更新最终内容，长度:', fullResponse.length);
                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                        }
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });
                    
                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.trim() === '') continue;
                        
                        console.log('[AI Ops SSE] 收到行:', line);
                        
                        // 解析 SSE 格式
                        if (line.startsWith('id:')) {
                            continue;
                        } else if (line.startsWith('event:')) {
                            currentEvent = line.substring(6).trim();
                            console.log('[AI Ops SSE] 事件类型:', currentEvent);
                            continue;
                        } else if (line.startsWith('data:')) {
                            const rawData = line.substring(5).trim();
                            console.log('[AI Ops SSE] 数据:', rawData, ', currentEvent:', currentEvent);
                            
                            // 解析可能包含多个JSON对象的数据
                            const processJsonMessages = (data) => {
                                const jsonPattern = /\{"type"\s*:\s*"[^"]+"\s*,\s*"data"\s*:\s*(?:"[^"]*"|null)\}/g;
                                const matches = data.match(jsonPattern);
                                
                                if (matches && matches.length > 0) {
                                    console.log('[AI Ops SSE] 匹配到', matches.length, '个JSON对象');
                                    for (const jsonStr of matches) {
                                        try {
                                            const sseMessage = JSON.parse(jsonStr);
                                            if (sseMessage.type === 'content') {
                                                fullResponse += sseMessage.data || '';
                                            } else if (sseMessage.type === 'plan') {
                                                // 处理计划创建事件
                                                const planText = `\n\n## 执行计划\n${sseMessage.message}\n\n`;
                                                fullResponse += planText;
                                            } else if (sseMessage.type === 'step_complete') {
                                                // 处理步骤完成事件
                                                const stepText = `\n完成 ${sseMessage.message}\n`;
                                                fullResponse += stepText;
                                            } else if (sseMessage.type === 'status') {
                                                // 处理状态更新事件
                                                const statusText = `\n进行中 ${sseMessage.message}\n`;
                                                fullResponse += statusText;
                                            } else if (sseMessage.type === 'report') {
                                                // 处理最终报告事件 - 流式输出
                                                console.log('AI Ops 最终报告生成');
                                                const reportText = `\n\n## 诊断报告\n\n${sseMessage.report || ''}\n`;
                                                fullResponse += reportText;
                                            } else if (sseMessage.type === 'complete') {
                                                // 处理完成事件
                                                console.log('AI Ops 诊断完成');
                                                const finalReport = (sseMessage.diagnosis && sseMessage.diagnosis.report)
                                                    || sseMessage.response
                                                    || '';
                                                if (finalReport && !fullResponse.includes(finalReport)) {
                                                    fullResponse += `\n\n${finalReport}`;
                                                }
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'done') {
                                                console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'error') {
                                                throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                            }
                                        } catch (e) {
                                            if (e.message.includes('智能运维')) throw e;
                                            console.log('[AI Ops SSE] 单个JSON解析失败:', jsonStr);
                                        }
                                    }
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                    return false;
                                }
                                return null;
                            };
                            
                            const result = processJsonMessages(rawData);
                            if (result === true) {
                                return; // 流结束
                            } else if (result === null) {
                                // 没有匹配到多个JSON，尝试单个JSON解析
                                try {
                                    const sseMessage = JSON.parse(rawData);
                                    if (sseMessage && sseMessage.type) {
                                        if (sseMessage.type === 'content') {
                                            fullResponse += sseMessage.data || '';
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'plan') {
                                            // 处理计划创建事件
                                            const planText = `\n\n## 执行计划\n${sseMessage.message}\n\n`;
                                            fullResponse += planText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'step_complete') {
                                            // 处理步骤完成事件
                                            const stepText = `\n完成 ${sseMessage.message}\n`;
                                            fullResponse += stepText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'status') {
                                            // 处理状态更新事件
                                            const statusText = `\n进行中 ${sseMessage.message}\n`;
                                            fullResponse += statusText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'report') {
                                            console.log('AI Ops 最终报告生成，流式输出中...');
                                            const reportText = `\n\n## 诊断报告\n\n${sseMessage.report || ''}\n`;
                                            fullResponse += reportText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'complete') {
                                            // 处理完成事件（后端字段为 diagnosis.report，兼容 response）
                                            console.log('AI Ops 诊断完成，最终内容长度:', fullResponse.length);
                                            const finalReport = (sseMessage.diagnosis && sseMessage.diagnosis.report)
                                                || sseMessage.response
                                                || '';
                                            if (finalReport && !fullResponse.includes(finalReport)) {
                                                fullResponse += `\n\n${finalReport}`;
                                            }
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'done') {
                                            console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'error') {
                                            throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                        }
                                    } else {
                                        fullResponse += rawData;
                                        if (loadingMessageElement) {
                                            this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                        }
                                    }
                                } catch (e) {
                                    if (e.message.includes('智能运维')) throw e;
                                    // 非 JSON 格式，直接追加原始数据
                                    fullResponse += rawData;
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            throw error;
        }
    }

    // 更新智能运维流式内容（实时显示）
    updateAIOpsStreamContent(messageElement, content) {
        if (!messageElement) return;
        
        // 添加 aiops-message 类
        messageElement.classList.add('aiops-message');
        
        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (messageContentWrapper) {
            let messageContent = messageContentWrapper.querySelector('.message-content');
            if (!messageContent) {
                messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContentWrapper.appendChild(messageContent);
            }
            // 流式显示时使用纯文本
            messageContent.textContent = content;
            this.scrollToBottom();
        }
    }

    // 更新智能运维消息（带折叠详情）
    updateAIOpsMessage(messageElement, response, details) {
        console.log('updateAIOpsMessage 被调用');
        console.log('messageElement:', messageElement);
        console.log('response:', response);
        console.log('response length:', response ? response.length : 0);
        console.log('details:', details);
        
        if (!messageElement) {
            // 如果没有传入消息元素，则创建新消息
            console.log('messageElement 为空，创建新消息');
            return this.addAIOpsMessage(response, details);
        }

        // 添加aiops-message类
        messageElement.classList.add('aiops-message');

        // 获取消息内容包装器
        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (!messageContentWrapper) {
            console.error('未找到 message-content-wrapper');
            return;
        }

        // 清空现有内容（保留消息内容容器）
        const messageContent = messageContentWrapper.querySelector('.message-content');
        if (!messageContent) {
            console.error('未找到 message-content');
            return;
        }

        // 移除加载动画相关的类和内容
        messageContent.classList.remove('loading-message-content');
        messageContent.textContent = '';
        
        // 移除加载图标（如果存在）
        const loadingIcon = messageContent.querySelector('.loading-spinner-icon');
        if (loadingIcon) {
            loadingIcon.remove();
        }

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            // 检查是否已存在详情容器
            let detailsContainer = messageElement.querySelector('.aiops-details');
            if (!detailsContainer) {
                detailsContainer = document.createElement('div');
                detailsContainer.className = 'aiops-details';
                messageContentWrapper.insertBefore(detailsContainer, messageContent);
            } else {
                // 清空现有详情
                detailsContainer.innerHTML = '';
            }

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';
            
            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
        }

        // 更新主要响应内容（使用Markdown渲染）
        console.log('开始渲染 Markdown');
        const renderedHtml = this.renderMarkdown(response);
        console.log('Markdown 渲染完成，HTML 长度:', renderedHtml ? renderedHtml.length : 0);
        messageContent.innerHTML = renderedHtml;
        console.log('innerHTML 已设置');
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        console.log('代码块高亮完成');
        
        // 写入当前会话，并立即同步到侧边栏历史（避免重复 push）
        this.persistAIOpsResult(response);
        
        this.scrollToBottom();
        return messageElement;
    }

    // 将 AIOps 结果写入当前会话并刷新侧边栏历史
    persistAIOpsResult(response) {
        if (!response || !String(response).trim()) {
            return;
        }

        this._isAIOpsSession = true;

        // 保证有一条用户侧标题消息，便于历史列表展示
        const hasUser = this.currentChatHistory.some(msg => msg.type === 'user');
        if (!hasUser) {
            this.currentChatHistory.unshift({
                type: 'user',
                content: 'AIOps 智能运维诊断',
                timestamp: new Date().toISOString()
            });
        }

        const lastAssistantIdx = [...this.currentChatHistory]
            .map((msg, idx) => (msg.type === 'assistant' ? idx : -1))
            .filter(idx => idx >= 0)
            .pop();

        if (lastAssistantIdx !== undefined) {
            this.currentChatHistory[lastAssistantIdx] = {
                type: 'assistant',
                content: response,
                timestamp: new Date().toISOString()
            };
        } else {
            this.currentChatHistory.push({
                type: 'assistant',
                content: response,
                timestamp: new Date().toISOString()
            });
        }

        // 立即出现在侧边栏；已存在则更新内容
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) {
            this.saveCurrentChat();
        } else {
            this.updateCurrentChatHistory();
            this.chatHistories[existingIndex].type = 'aiops';
            const firstUser = this.currentChatHistory.find(msg => msg.type === 'user');
            if (firstUser) {
                this.chatHistories[existingIndex].title = firstUser.content.substring(0, 30);
            }
            this.saveChatHistories();
        }
        this._aiopsHistorySaved = true;
        this.renderChatHistory();
    }

    // 添加智能运维消息（带折叠详情）- 保留用于兼容性
    addAIOpsMessage(response, details) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant aiops-message';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = this.createAgentMark();
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            const detailsContainer = document.createElement('div');
            detailsContainer.className = 'aiops-details';

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';
            
            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
            messageContentWrapper.appendChild(detailsContainer);
        }

        // 主要响应内容 - 后显示（使用Markdown渲染）
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.renderMarkdown(response);
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);
        
        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
        }

        return messageDiv;
    }

    createAgentMark() {
        return '<span class="agent-mark" aria-hidden="true"><span></span><span></span></span>';
    }

    // HTML转义
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 触发智能运维（点击智能运维按钮时直接调用）
    async triggerAIOps() {
        if (this.isStreaming) {
            this.showNotification('请等待当前操作完成', 'warning');
            return;
        }

        // 新建对话（会先保存上一轮）
        this.newChat();
        this._isAIOpsSession = true;
        this._aiopsHistorySaved = false;

        // 写入用户意图，保证侧边栏有可读标题
        const aiopsTitle = `AIOps 诊断 ${new Date().toLocaleString('zh-CN', {
            month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
        })}`;
        this.addMessage('user', aiopsTitle);
        
        // 添加"分析中..."的消息（带旋转动画）
        const loadingMessage = this.addLoadingMessage('分析中...');
        this.currentAIOpsMessage = loadingMessage; // 保存消息引用用于后续更新
        
        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            await this.sendAIOpsRequest(loadingMessage);
            // 兜底：流正常结束但未触发 complete 时，仍尝试落盘
            if (!this._aiopsHistorySaved && this.currentChatHistory.length > 0) {
                const lastAssistant = [...this.currentChatHistory].reverse()
                    .find(msg => msg.type === 'assistant');
                if (lastAssistant && lastAssistant.content) {
                    this.persistAIOpsResult(lastAssistant.content);
                }
            }
        } catch (error) {
            console.error('智能运维分析失败:', error);
            // 更新消息为错误信息
            if (loadingMessage) {
                const messageContent = loadingMessage.querySelector('.message-content');
                if (messageContent) {
                    messageContent.textContent = '抱歉，智能运维分析时出现错误：' + error.message;
                }
            }
            this.persistAIOpsResult('抱歉，智能运维分析时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.currentAIOpsMessage = null;
            this.updateUI();
        }
    }

    // 显示/隐藏加载遮罩层
    showLoadingOverlay(show) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                // 更新文字为智能运维
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '智能运维分析中，请稍候...';
                if (loadingSubtext) loadingSubtext.textContent = '后端正在处理，请耐心等待';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }

    // 显示/隐藏上传遮罩层
    showUploadOverlay(show, fileName = '') {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                // 更新文字为上传中
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '正在上传文件...';
                if (loadingSubtext) loadingSubtext.textContent = fileName ? `上传: ${fileName}` : '请稍候';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new IntelliOpsAgentApp();
});
