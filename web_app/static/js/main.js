document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const spinner = submitBtn.querySelector('.spinner-border');
    const statusContainer = document.getElementById('status-container');
    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const resultContainer = document.getElementById('result-container');
    const resultImage = document.getElementById('result-image');
    const errorContainer = document.getElementById('error-container');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // --- 重置UI ---
        submitBtn.disabled = true;
        spinner.classList.remove('d-none');
        statusContainer.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        updateProgress(0, "正在上传图像...");

        const formData = new FormData(form);

        try {
            // --- 1. 提交任务 ---
            const response = await fetch('/run_stereo', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '上传失败');
            }

            const data = await response.json();
            const statusUrl = data.status_url;

            // --- 2. 开始轮询任务状态 ---
            pollStatus(statusUrl);

        } catch (error) {
            showError(error.message);
        }
    });

    function pollStatus(statusUrl) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(statusUrl);
                const data = await response.json();

                updateProgress(data.progress, data.status);

                if (data.state === 'SUCCESS') {
                    clearInterval(interval);
                    showResult(data.result.result_image);
                } else if (data.state === 'FAILURE') {
                    clearInterval(interval);
                    showError(`任务失败: ${data.status}`);
                }
            } catch (error) {
                clearInterval(interval);
                showError('轮询状态时发生网络错误');
            }
        }, 2000); // 每2秒查询一次
    }

    function updateProgress(progress, message) {
        statusMessage.textContent = message;
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressBar.textContent = `${progress}%`;
    }

    function showResult(imageUrl) {
        resultContainer.classList.remove('d-none');
        resultImage.src = imageUrl;
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('d-none');
        statusContainer.classList.add('d-none');
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
    }
});