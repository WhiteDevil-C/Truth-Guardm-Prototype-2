// TruthGuard: remember the user's latest text selection on this page.
function saveTruthGuardSelection() {
    const selectedText = window.getSelection()?.toString().trim();

    if (!selectedText) return;

    chrome.storage.local.set({
        truthguardSelection: {
            text: selectedText,
            url: window.location.href,
            updatedAt: Date.now()
        }
    });
}

document.addEventListener("mouseup", saveTruthGuardSelection, true);
document.addEventListener("keyup", saveTruthGuardSelection, true);
