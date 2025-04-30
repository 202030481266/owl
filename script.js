// 检测系统主题偏好
function detectSystemTheme() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }
    return 'light';
}

// 修改初始化函数
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const systemTheme = detectSystemTheme();

    // 优先使用保存的主题，否则使用系统主题
    const themeToUse = savedTheme || systemTheme;

    document.documentElement.setAttribute('data-theme', themeToUse);
    themeSwitcher.textContent = themeToUse === 'dark' ? '☀️' : '🌙';
    return themeToUse;
}

// 主题切换功能
const themeSwitcher = document.getElementById('theme-switcher');

// 初始化主题
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeSwitcher.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    return savedTheme;
}

let currentTheme = initTheme();

// 切换主题
themeSwitcher.addEventListener('click', () => {
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    themeSwitcher.textContent = newTheme === 'dark' ? '☀️' : '🌙';
    currentTheme = newTheme;  // 更新当前主题状态
});

// 修改后的 fetchMarkdownFiles 函数
async function fetchMarkdownFiles() {
    try {
        const response = await fetch('/file-list.json');
        if (!response.ok) throw new Error('无法获取文件列表');
        return await response.json();
    } catch (error) {
        console.error('获取文件列表失败:', error);
        return [];
    }
}

// 修改后的 loadMarkdownFile 函数
async function loadMarkdownFile(filePath, fileName) {
    try {
        const response = await fetch(filePath);  // 现在路径已经是服务器上的路径
        if (!response.ok) throw new Error('无法加载文件');
        const markdownText = await response.text();

        document.getElementById('current-filename').textContent = fileName;
        const htmlContent = marked.parse(markdownText);
        document.getElementById('markdown-content').innerHTML = htmlContent;

        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        const fileItems = document.querySelectorAll('.file-list li');
        fileItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.path === filePath) {
                item.classList.add('active');
            }
        });
    } catch (error) {
        console.error('加载Markdown文件失败:', error);
        document.getElementById('markdown-content').innerHTML = `
            <div class="error-message">
                <p>无法加载文件: ${fileName}</p>
                <p>${error.message}</p>
            </div>
        `;
    }
}

// 渲染文件列表
async function renderFileList() {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';

    const files = await fetchMarkdownFiles();

    files.forEach(file => {
        const li = document.createElement('li');
        li.textContent = file.name;
        li.dataset.path = file.path;
        li.addEventListener('click', () => loadMarkdownFile(file.path, file.name));
        fileList.appendChild(li);
    });
}

function setupResizable() {
    const sidebar = document.getElementById('file-sidebar');
    const resizeHandle = document.getElementById('resize-handle');
    const mainContent = document.querySelector('.main-content');

    let isResizing = false;
    let lastDownX = 0;

    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        lastDownX = e.clientX;
        document.body.style.cursor = 'col-resize';
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        e.preventDefault();
    });

    function handleMouseMove(e) {
        if (!isResizing) return;

        const offsetRight = mainContent.getBoundingClientRect().right - e.clientX;
        const newWidth = e.clientX - mainContent.getBoundingClientRect().left;

        // 应用最小和最大宽度限制
        const minWidth = 200;
        const maxWidth = mainContent.clientWidth * 0.5;

        if (newWidth > minWidth && newWidth < maxWidth) {
            sidebar.style.width = `${newWidth}px`;
        }
    }

    function handleMouseUp() {
        isResizing = false;
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);

        // 保存宽度到本地存储
        localStorage.setItem('sidebarWidth', sidebar.style.width);
    }

    // 恢复上次保存的宽度
    const savedWidth = localStorage.getItem('sidebarWidth');
    if (savedWidth) {
        sidebar.style.width = savedWidth;
    }
}

// 刷新文件列表
document.getElementById('refresh-files').addEventListener('click', renderFileList);

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    setupResizable();
    renderFileList();

    // 默认加载第一个文件
    setTimeout(() => {
        const firstFile = document.querySelector('.file-list li');
        if (firstFile) {
            firstFile.click();
        }
    }, 300);
});