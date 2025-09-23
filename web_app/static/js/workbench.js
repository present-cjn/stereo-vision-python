// web_app/static/js/workbench.js

document.addEventListener('DOMContentLoaded', () => {
    console.log("工作台界面已加载！");

    // --- 1. 获取所有需要操作的界面元素 ---
    // 按钮
    const uploadBtn = document.getElementById('upload-btn');
    const rectifyBtn = document.getElementById('rectify-btn');
    const matchBtn = document.getElementById('match-btn');
    // 文件输入框
    const leftImageInput = document.getElementById('left-image-input');
    const rightImageInput = document.getElementById('right-image-input');
    const paramsFileInput = document.getElementById('params-file-input');
    // 步骤容器
    const step2Container = document.getElementById('step-2-rectify');
    const step3Container = document.getElementById('step-3-match');
    // 预览区域
    const previewArea = document.getElementById('preview-area');
    // 存储当前任务的ID，以便在不同步骤间传递
    let currentTaskId = null;

    // --- 2. 为“上传并预览”按钮绑定点击事件 ---
    uploadBtn.addEventListener('click', async () => {
        // a. 获取用户选择的文件
        const leftFile = leftImageInput.files[0];
        const rightFile = rightImageInput.files[0];
        const paramsFile = paramsFileInput.files[0];

        if (!leftFile || !rightFile || !paramsFile) {
            alert('请确保已选择左图像、右图像和标定文件！');
            return;
        }

        // b. 准备表单数据
        const formData = new FormData();
        formData.append('left_image', leftFile);
        formData.append('right_image', rightFile);
        formData.append('params_file', paramsFile);

        // c. 显示加载状态
        uploadBtn.disabled = true;
        uploadBtn.textContent = '正在上传...';

        try {
            // d. 发送异步请求到后端
            const response = await fetch('/upload_images', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '上传失败，未知错误。');
            }

            // e. 请求成功，处理返回的数据
            console.log('上传成功:', data);
            currentTaskId = data.task_id; // 保存任务ID

            // 更新预览区域的图像
            updatePreviewImages(data.left_image_url, data.right_image_url);

            // 解锁下一步
            step2Container.classList.remove('d-none');
            alert('文件上传成功！现在可以进行下一步：立体校正。');

        } catch (error) {
            console.error('上传时发生错误:', error);
            alert(`错误: ${error.message}`);
        } finally {
            // f. 恢复按钮状态
            uploadBtn.disabled = false;
            uploadBtn.textContent = '上传并预览';
        }
    });

    // --- 3. 为“开始校正”按钮绑定点击事件 ---
    rectifyBtn.addEventListener('click', async () => {
        if (!currentTaskId) {
            alert('错误：没有有效的任务ID。请先上传图片。');
            return;
        }

        rectifyBtn.disabled = true;
        rectifyBtn.textContent = '正在校正...';

        try {
            // a. 发送请求到新的校正接口
            const response = await fetch('/rectify_images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ task_id: currentTaskId })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error);

            // b. 请求成功，用返回的新图片URL更新预览区
            console.log('校正成功:', data);
            updatePreviewImages(data.left_rectified_url, data.right_rectified_url, true);

            // c. 解锁下一步
            step3Container.classList.remove('d-none');
            alert('立体校正成功！现在可以进行下一步：立体匹配。');

        } catch (error) {
            console.error('校正时发生错误:', error);
            alert(`错误: ${error.message}`);
        } finally {
            rectifyBtn.disabled = false;
            rectifyBtn.textContent = '开始校正';
        }
    });


    function updatePreviewImages(leftUrl, rightUrl, isRectified = false) {
        // 清空预览区的占位符
        previewArea.innerHTML = '';

        // 创建并添加左图像
        const leftImg = document.createElement('img');
        leftImg.src = leftUrl;
        leftImg.className = 'preview-image';
        previewArea.appendChild(leftImg);

        // 创建并添加右图像
        const rightImg = document.createElement('img');
        rightImg.src = rightUrl;
        rightImg.className = 'preview-image';
        previewArea.appendChild(rightImg);

        // 动态修改样式来画线或移除线
        // 查找文档中是否已有我们添加的样式
        let styleTag = document.getElementById('rectify-style');
        if (!styleTag) {
            styleTag = document.createElement('style');
            styleTag.id = 'rectify-style';
            document.head.appendChild(styleTag);
        }

        if (isRectified) {
            // 如果是校正后的图像，我们添加带水平线的背景
            styleTag.innerHTML = `
                .preview-image {
                    max-width: 48%;
                    max-height: 95%;
                    object-fit: contain;
                    background-image: repeating-linear-gradient(to bottom, rgba(128, 128, 128, 0.5) 0, rgba(128, 128, 128, 0.5) 1px, transparent 1px, transparent 20px);
                }
            `;
        } else {
             // 如果是原始图像，确保没有背景线
            styleTag.innerHTML = `
                .preview-image {
                    max-width: 48%;
                    max-height: 95%;
                    object-fit: contain;
                }
            `;
        }
    }
});