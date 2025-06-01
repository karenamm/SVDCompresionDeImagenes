// svdapp/static/svdapp/js/scripts.js

document.addEventListener('DOMContentLoaded', function() {
    const processSelect = document.getElementById('id_process_type');
    const patchGroup = document.getElementById('patch-size-group');

    function togglePatchSize() {
        if (processSelect.value === 'pca_patches') {
            patchGroup.style.display = 'block';
        } else {
            patchGroup.style.display = 'none';
            const patchInput = document.getElementById('id_patch_size');
            if (patchInput) {
                patchInput.value = '';
            }
        }
    }

    togglePatchSize();
    processSelect.addEventListener('change', togglePatchSize);
});
