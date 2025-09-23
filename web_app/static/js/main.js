document.addEventListener('DOMContentLoaded', () => {
    // --- 获取所有UI元素 ---
    const recoForm = document.getElementById('reco-form');
    const calibForm = document.getElementById('calib-form');
    const recoSubmitBtn = document.getElementById('reco-submit-btn');
    const calibSubmitBtn = document.getElementById('calib-submit-btn');

    const statusContainer = document.getElementById('status-container');
    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');

    // --- 为重建表单绑定事件 ---
    recoForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(recoForm);
        startTask('/run_stereo', formData, recoSubmitBtn, handleRecoResult);
    });

    // --- 为标定表单绑定事件 ---
    calibForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(calibForm);
        startTask('/start_calibration', formData, calibSubmitBtn, handleCalibResult);
    });

    async function startTask(url, formData, submitBtn, resultHandler) {
        const spinner = submitBtn.querySelector('.spinner-border') || document.createElement('span');

        // --- 重置UI ---
        submitBtn.disabled = true;
        if (spinner) spinner.classList.remove('d-none');
        statusContainer.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        updateProgress(0, "正在上传文件...");

        try {
            const response = await fetch(url, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '请求失败');
            }
            const data = await response.json();
            pollStatus(data.status_url, submitBtn, resultHandler);
        } catch (error) {
            showError(error.message, submitBtn);
        }
    }

    function pollStatus(statusUrl, submitBtn, resultHandler) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(statusUrl);
                const data = await response.json();
                updateProgress(data.progress, data.status);

                if (data.state === 'SUCCESS') {
                    clearInterval(interval);
                    resultHandler(data.result);
                    resetSubmitBtn(submitBtn);
                } else if (data.state === 'FAILURE') {
                    clearInterval(interval);
                    showError(`任务失败: ${data.status}`, submitBtn);
                }
            } catch (error) {
                clearInterval(interval);
                showError('轮询状态时发生网络错误', submitBtn);
            }
        }, 2000);
    }

    function handleRecoResult(result) {
        resultContainer.innerHTML = `
            <h2 class="mb-4">重建结果：深度图</h2>
            <div class="result-container d-flex justify-content-center align-items-center bg-light border rounded">
                <img src="${result.result_image}" alt="结果图像" class="img-fluid">
            </div>`;
        resultContainer.classList.remove('d-none');
    }

    function handleCalibResult(result) {
        resultContainer.innerHTML = `
            <h2 class="mb-4">标定结果</h2>
            <div class="alert alert-success">
                <h4 class="alert-heading">标定成功！</h4>
                <p>双目相机的几何参数已成功计算并保存。</p>
                <hr>
                <p class="mb-0"><strong>最终重投影误差: ${result.reprojection_error}</strong> (一个好的标定结果此值应小于1.0)</p>
                <a href="${result.download_url}" class="btn btn-info mt-3">下载标定文件 (stereo_params.yml)</a>
            </div>`;
        resultContainer.classList.remove('d-none');
    }

    function updateProgress(progress, message) {
        statusMessage.textContent = message;
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressBar.textContent = `${progress}%`;
    }

    function showError(message, submitBtn) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('d-none');
        statusContainer.classList.add('d-none');
        resetSubmitBtn(submitBtn);
    }

    function resetSubmitBtn(submitBtn) {
        const spinner = submitBtn.querySelector('.spinner-border');
        submitBtn.disabled = false;
        if(spinner) spinner.classList.add('d-none');
    }
});
