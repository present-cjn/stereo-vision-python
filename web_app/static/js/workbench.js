// web_app/static/js/workbench.js

document.addEventListener('DOMContentLoaded', () => {
    console.log("工作台界面已加载！");

    // --- 1. 获取所有UI元素 ---
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
    const step4Container = document.getElementById('step-4-reconstruct');
    const reconstructBtn = document.getElementById('reconstruct-btn');
    const resultsInfoContainer = document.getElementById('results-info-container');

    // 动态图片预览区
    const pipelineArea = document.getElementById('pipeline-area');

    // 右侧SGBM参数面板
    const sgbmParamsContainer = document.getElementById('sgbm-params-container');
    const sgbmSliders = document.querySelectorAll('.sgbm-param');

    // 存储当前任务的状态
    let currentTaskId = null;
    let rectifiedLeftUrl = null;

    // --- 2. 为“上传”按钮绑定事件 ---
    uploadBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('left_image', leftImageInput.files[0]);
        formData.append('right_image', rightImageInput.files[0]);
        formData.append('params_file', paramsFileInput.files[0]);
        if (!leftImageInput.files[0] || !rightImageInput.files[0] || !paramsFileInput.files[0]) {
            alert('请确保已选择所有三个文件！'); return;
        }
        await performAction(uploadBtn, '/upload_data', { method: 'POST', body: formData }, (data) => {
            currentTaskId = data.task_id;

            // 动态创建并插入新的图片行
            createAndInsertImageRow(
                'original-images',
                '1. 原始图像',
                data.left_image_url,
                data.right_image_url
            );

            // 隐藏可能存在的旧结果，重置工作台状态
            // 由于现在是动态插入，这里只需要清空之前的任何内容
            // ... (如果需要，可以在这里清空 pipelineArea.innerHTML)

            // 解锁下一步
            step2Container.classList.remove('d-none');
            alert('文件上传成功！现在可以进行下一步：立体校正。');
        }, "正在上传...", "上传并预览");
    });

    // --- 3. 为“校正”按钮绑定事件 ---
    rectifyBtn.addEventListener('click', async () => {
        if (!currentTaskId) {
            alert('错误：没有有效的任务ID。请先上传图片。');
            return;
        }
        await performAction(rectifyBtn, '/rectify_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId })
        }, (data) => {
            rectifiedLeftUrl = data.left_rectified_url;

            // 动态创建并插入新的图片行
            createAndInsertImageRow(
                'rectified-images',
                '2. 校正后图像 (已行对准)',
                data.left_rectified_url,
                data.right_rectified_url,
                true // 添加网格线
            );

            // 解锁下一步和参数面板
            step3Container.classList.remove('d-none');
            sgbmParamsContainer.classList.remove('d-none');
            alert('立体校正成功！现在可以调整参数并生成视差图。');
        }, "正在校正...", "开始校正");
    });

    // --- 4. 为“匹配”按钮和滑块绑定事件 ---
    matchBtn.addEventListener('click', updateDisparityMap);
    sgbmSliders.forEach(slider => {
        const valueSpan = document.getElementById(`${slider.id}Value`);
        // 当拖动滑块时，实时更新旁边的数值显示
        slider.addEventListener('input', () => {
            if (valueSpan) valueSpan.textContent = slider.value;
        });
        // 当拖动结束时（松开鼠标），触发视差图的重新计算
        slider.addEventListener('change', updateDisparityMap);
    });

    // 为“开始重建”按钮绑定事件
    reconstructBtn.addEventListener('click', async () => {
        if (!currentTaskId) {
            alert('错误：请先完成前面的所有步骤。');
            return;
        }
        await performAction(reconstructBtn, '/reconstruct_3d', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId })
        }, (data) => {
            console.log('重建成功:', data);

            // 在右侧面板显示结果信息
            resultsInfoContainer.innerHTML = `
                <div class="alert alert-success">
                    <h5 class="alert-heading">重建完成！</h5>
                    <p>成功生成点云，包含 <strong>${data.point_count}</strong> 个三维点。</p>
                    <hr>
                    <a href="${data.point_cloud_url}" class="btn btn-info w-100" download>下载点云文件 (.ply)</a>
                </div>
            `;
        }, "正在重建...", "开始重建");
    });

    async function updateDisparityMap() {
        if (!currentTaskId) {
            alert('错误：请先上传并校正图像。');
            return;
        }
        const params = {};
        sgbmSliders.forEach(slider => {
            params[slider.id] = parseInt(slider.value, 10);
        });

        await performAction(matchBtn, '/compute_disparity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId, params: params })
        }, (data) => {
            // 动态创建并插入新的图片行
            createAndInsertImageRow(
                'disparity-images',
                '3. 视差图',
                rectifiedLeftUrl,
                data.disparity_map_url
            );

            step4Container.classList.remove('d-none');
        }, "正在计算...", "生成/更新视差图");
    }

    // --- 辅助函数 ---
    async function performAction(button, url, options, onSuccess, loadingText = '处理中...', defaultText = '执行') {
        button.disabled = true;
        button.innerHTML = `<span class="spinner-border spinner-border-sm"></span> ${loadingText}`;
        try {
            const response = await fetch(url, options);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '请求失败');
            onSuccess(data);
        } catch (error) {
            alert(`错误: ${error.message}`);
        } finally {
            button.disabled = false;
            button.textContent = defaultText;
        }
    }

    // 新的动态创建和插入图片行的函数
    function createAndInsertImageRow(id, title, leftUrl, rightUrl, addGrid = false) {
        // 创建主容器
        const newRow = document.createElement('div');
        newRow.id = id;
        newRow.className = 'preview-row';

        // 创建标题
        const titleElement = document.createElement('h6');
        titleElement.textContent = title;

        // 创建图片对容器
        const imagePairContainer = document.createElement('div');
        imagePairContainer.className = 'image-pair-container';

        // 创建左侧图片占位符
        const leftPlaceholder = document.createElement('div');
        leftPlaceholder.className = 'image-placeholder';
        const leftImg = document.createElement('img');
        leftImg.src = leftUrl;
        leftPlaceholder.appendChild(leftImg);

        // 创建右侧图片占位符
        const rightPlaceholder = document.createElement('div');
        rightPlaceholder.className = 'image-placeholder';
        const rightImg = document.createElement('img');
        rightImg.src = rightUrl;
        rightPlaceholder.appendChild(rightImg);

        // 如果需要，添加网格背景
        if (addGrid) {
            const bgStyle = "repeating-linear-gradient(to bottom, rgba(128, 128, 128, 0.5) 0, rgba(128, 128, 128, 0.5) 1px, transparent 1px, transparent 20px)";
            leftPlaceholder.style.backgroundImage = bgStyle;
            rightPlaceholder.style.backgroundImage = bgStyle;
        }

        // 组装所有元素
        imagePairContainer.appendChild(leftPlaceholder);
        imagePairContainer.appendChild(rightPlaceholder);
        newRow.appendChild(titleElement);
        newRow.appendChild(imagePairContainer);

        // 插入到流水线区域的最顶部
        pipelineArea.prepend(newRow);
    }
});