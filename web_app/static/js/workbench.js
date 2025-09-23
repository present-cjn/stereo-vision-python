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
    // 存储任务ID
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

    function updatePreviewImages(leftUrl, rightUrl) {
        // 清空预览区的占位符
        previewArea.innerHTML = ''; 
        
        // 创建并添加左图像
        const leftImg = document.createElement('img');
        leftImg.src = leftUrl;
        leftImg.style.maxWidth = '48%';
        leftImg.style.maxHeight = '95%';
        leftImg.style.objectFit = 'contain';
        previewArea.appendChild(leftImg);

        // 创建并添加右图像
        const rightImg = document.createElement('img');
        rightImg.src = rightUrl;
        rightImg.style.maxWidth = '48%';
        rightImg.style.maxHeight = '95%';
        rightImg.style.objectFit = 'contain';
        previewArea.appendChild(rightImg);
    }
});