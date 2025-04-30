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
        
        // 配置marked选项以增强渲染
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,               // 添加换行符支持
            gfm: true,                  // 使用GitHub风格Markdown
            headerIds: true,            // 为标题添加ID
            mangle: false,              // 不转义标题文本
            smartLists: true,           // 使用更智能的列表行为
            smartypants: true,          // 使用更智能的标点符号
            xhtml: false                // 不使用自闭合标签
        });
        
        const htmlContent = marked.parse(markdownText);
        document.getElementById('markdown-content').innerHTML = htmlContent;

        // 应用代码高亮
        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        // 为图片添加点击放大功能
        document.querySelectorAll('#markdown-content img').forEach(img => {
            img.addEventListener('click', function() {
                this.classList.toggle('fullscreen');
            });
            img.style.cursor = 'pointer';
        });

        // 为标题添加锚点链接
        document.querySelectorAll('#markdown-content h1, #markdown-content h2, #markdown-content h3').forEach(heading => {
            const id = heading.textContent.toLowerCase().replace(/\s+/g, '-').replace(/[^\w\-]+/g, '');
            heading.id = id;
            
            const anchor = document.createElement('a');
            anchor.className = 'header-anchor';
            anchor.href = `#${id}`;
            anchor.textContent = '#';
            anchor.title = '链接到此标题';
            heading.appendChild(anchor);
        });

        // 生成目录
        generateTOC();

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

// 生成目录功能
function generateTOC() {
    const tocList = document.getElementById('toc-list');
    tocList.innerHTML = '';
    
    const headings = document.querySelectorAll('#markdown-content h1, #markdown-content h2, #markdown-content h3');
    
    if (headings.length === 0) {
        tocList.innerHTML = '<li>本文档没有标题</li>';
        return;
    }
    
    headings.forEach(heading => {
        const li = document.createElement('li');
        li.className = `toc-${heading.tagName.toLowerCase()}`;
        
        const a = document.createElement('a');
        a.href = `#${heading.id}`;
        a.textContent = heading.textContent.replace(/#$/, ''); // 移除锚点字符
        a.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
        
        li.appendChild(a);
        tocList.appendChild(li);
    });
}

// 切换目录显示
document.getElementById('toc-toggle').addEventListener('click', function() {
    const tocContainer = document.getElementById('toc-container');
    tocContainer.classList.toggle('visible');
    
    // 如果目录为空，则生成目录
    if (tocContainer.classList.contains('visible') && document.getElementById('toc-list').children.length === 0) {
        generateTOC();
    }
});

// 打印功能
document.getElementById('print-content').addEventListener('click', function() {
    const fileName = document.getElementById('current-filename').textContent;
    const contentToPrint = document.getElementById('markdown-content').innerHTML;
    
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>打印: ${fileName}</title>
            <link rel="stylesheet" href="styles.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
            <style>
                body {
                    padding: 20px;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                @media print {
                    .header-anchor {
                        display: none;
                    }
                }
            </style>
        </head>
        <body>
            <h1>${fileName}</h1>
            <div class="markdown-content">${contentToPrint}</div>
        </body>
        </html>
    `);
    printWindow.document.close();
    
    // 等待样式加载完成后打印
    setTimeout(() => {
        printWindow.print();
    }, 500);
});

// 文件搜索功能
document.getElementById('file-search').addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    const fileItems = document.querySelectorAll('.file-list li:not(.loading):not(.error):not(.no-files)');
    
    fileItems.forEach(item => {
        const fileName = item.textContent.toLowerCase();
        if (fileName.includes(searchTerm)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
    
    // 显示没有匹配结果的提示
    const visibleCount = Array.from(fileItems).filter(item => item.style.display !== 'none').length;
    const noResultsItem = document.querySelector('.file-list li.no-results');
    
    if (visibleCount === 0 && searchTerm !== '') {
        if (!noResultsItem) {
            const li = document.createElement('li');
            li.className = 'no-results';
            li.textContent = `没有匹配 "${searchTerm}" 的文件`;
            document.getElementById('file-list').appendChild(li);
        } else {
            noResultsItem.textContent = `没有匹配 "${searchTerm}" 的文件`;
            noResultsItem.style.display = 'block';
        }
    } else if (noResultsItem) {
        noResultsItem.style.display = 'none';
    }
});

// 渲染文件列表
async function renderFileList() {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '<li class="loading">加载文件列表中...</li>';

    try {
        const files = await fetchMarkdownFiles();
        
        fileList.innerHTML = '';
        
        if (files.length === 0) {
            fileList.innerHTML = '<li class="no-files">没有找到Markdown文件</li>';
            return;
        }

        // 根据文件名排序
        files.sort((a, b) => a.name.localeCompare(b.name));

        files.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file.name;
            li.title = file.name; // 添加悬停提示，显示完整文件名
            li.dataset.path = file.path;
            li.addEventListener('click', () => loadMarkdownFile(file.path, file.name));
            fileList.appendChild(li);
        });
    } catch (error) {
        fileList.innerHTML = `<li class="error">加载文件列表失败: ${error.message}</li>`;
    }
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