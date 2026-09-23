const API_BASE = "http://127.0.0.1:8000";


// =====================================================
// DOM ELEMENTS
// =====================================================

const browseBtn =
    document.getElementById("browseBtn");

const fileInput =
    document.getElementById("fileInput");

const uploadBox =
    document.getElementById("uploadBox");

const documentList =
    document.getElementById("documentList");

const documentCount =
    document.querySelector(".document-count");

const chatForm =
    document.getElementById("chatForm");

const sendBtn =
    document.getElementById("sendBtn");

const questionInput =
    document.getElementById("questionInput");

const chatMessages =
    document.getElementById("chatMessages");


// =====================================================
// APPLICATION STATE
// =====================================================

let currentDocumentId = null;
let currentChatId = null;


// =====================================================
// FILE UPLOAD
// =====================================================

browseBtn.addEventListener(
    "click",
    () => {
        fileInput.click();
    }
);


fileInput.addEventListener(
    "change",
    () => {
        uploadFiles([...fileInput.files]);
    }
);


uploadBox.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        uploadBox.classList.add(
            "dragging"
        );
    }
);


uploadBox.addEventListener(
    "dragleave",
    () => {

        uploadBox.classList.remove(
            "dragging"
        );
    }
);


uploadBox.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        uploadBox.classList.remove(
            "dragging"
        );

        uploadFiles(
            [...event.dataTransfer.files]
        );
    }
);


// =====================================================
// UPLOAD DOCUMENTS
// =====================================================

async function uploadFiles(files) {

    for (const file of files) {

        const form =
            new FormData();

        form.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    `${API_BASE}/api/documents/upload`,
                    {
                        method: "POST",
                        body: form
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data?.error?.message ||
                    "Upload failed"
                );
            }


            currentDocumentId =
                data.document_id;

            currentChatId = null;


            if (data.duplicate) {

                addAssistantMessage(
                    `${file.name} was already uploaded. Using the existing document.`
                );

            } else {

                addAssistantMessage(
                    `${file.name} uploaded. Processing has started.`
                );
            }


        } catch (error) {

            addAssistantMessage(
                `Upload error: ${escapeHTML(error.message)}`
            );
        }
    }


    fileInput.value = "";

    await loadDocuments();
}


// =====================================================
// LOAD DOCUMENTS
// =====================================================

async function loadDocuments() {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/documents`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data?.error?.message ||
                "Failed to load documents"
            );
        }


        const docs =
            data.documents || [];


        documentList.innerHTML = "";


        documentCount.textContent =
            `${docs.length} document${
                docs.length === 1
                    ? ""
                    : "s"
            }`;


        if (!docs.length) {

            documentList.innerHTML = `
                <div class="empty-documents">

                    <i class="bi bi-folder2-open"></i>

                    <p>
                        No documents uploaded yet.
                    </p>

                </div>
            `;

            return;
        }


        docs.forEach(
            (doc) => {

                const card =
                    document.createElement(
                        "div"
                    );


                card.className =
                    "document-card";


                const isPdf =
                    doc.filename
                        .toLowerCase()
                        .endsWith(".pdf");


                const ready =
                    doc.status === "indexed";


                card.innerHTML = `

                    <div class="file-icon ${
                        isPdf
                            ? "pdf"
                            : "doc"
                    }">

                        <i class="bi ${
                            isPdf
                                ? "bi-file-earmark-pdf"
                                : "bi-file-earmark-text"
                        }"></i>

                    </div>


                    <div class="document-info">

                        <h6>
                            ${escapeHTML(
                                doc.filename
                            )}
                        </h6>

                        <span>
                            ${formatBytes(
                                doc.size_bytes
                            )}
                            •
                            ${escapeHTML(
                                doc.status
                            )}
                        </span>

                    </div>


                    <div class="document-status ${
                        ready
                            ? "indexed"
                            : "processing"
                    }">

                        ${
                            ready

                                ? `
                                    <i class="bi bi-check-circle"></i>
                                    Indexed
                                  `

                                : `
                                    <span class="spinner-border spinner-border-sm"></span>
                                    ${escapeHTML(
                                        doc.status
                                    )}
                                  `
                        }

                    </div>
                `;


                // ---------------------------------------------
                // SELECT DOCUMENT
                // ---------------------------------------------

                card.addEventListener(
                    "click",
                    () => {

                        currentDocumentId =
                            doc.id;

                        currentChatId =
                            null;


                        addAssistantMessage(
                            `Selected ${escapeHTML(
                                doc.filename
                            )}. Start a new question.`
                        );
                    }
                );


                documentList.appendChild(
                    card
                );
            }
        );


        // ---------------------------------------------
        // POLL PROCESSING DOCUMENTS
        // ---------------------------------------------

        if (
            docs.some(
                (doc) =>
                    ["queued", "processing"]
                        .includes(doc.status)
            )
        ) {

            setTimeout(
                loadDocuments,
                1800
            );
        }


    } catch (error) {

        console.error(
            "Failed to load documents:",
            error
        );
    }
}


// =====================================================
// CHAT SEND BUTTON
// =====================================================

// IMPORTANT:
// There is no form submission anymore.
// The button directly calls sendMessage().

sendBtn.addEventListener(
    "click",
    () => {
        sendMessage();
    }
);


// =====================================================
// ENTER KEY
// =====================================================

questionInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


// =====================================================
// SEND MESSAGE
// =====================================================

async function sendMessage() {

    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    // ---------------------------------------------
    // CHECK DOCUMENT
    // ---------------------------------------------

    if (!currentDocumentId) {

        addAssistantMessage(
            "Please upload and select a document first."
        );

        return;
    }


    // ---------------------------------------------
    // SHOW USER MESSAGE
    // ---------------------------------------------

    addUserMessage(
        question
    );


    questionInput.value = "";

    sendBtn.disabled = true;


    try {

        // -----------------------------------------
        // CREATE REQUEST BODY
        // -----------------------------------------

        const requestBody = {

            question:
                question,

            document_id:
                currentDocumentId
        };


        // Only send chat_id when
        // an existing chat already exists.

        if (currentChatId) {

            requestBody.chat_id =
                currentChatId;
        }


        console.log(
            "Sending chat request:",
            requestBody
        );


        // -----------------------------------------
        // SEND TO FASTAPI
        // -----------------------------------------

        const response =
            await fetch(
                `${API_BASE}/api/chat`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            requestBody
                        )
                }
            );


        const data =
            await response.json();


        console.log(
            "Chat response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                data?.error?.message ||
                "Chat request failed"
            );
        }


        // -----------------------------------------
        // SAVE CHAT ID
        // -----------------------------------------

        currentChatId =
            data.chat_id;


        // -----------------------------------------
        // DISPLAY ANSWER
        // -----------------------------------------

        addAssistantMessage(
            data.answer,
            data.sources || [],
            data.cache_hit
        );


    } catch (error) {

        console.error(
            "Chat error:",
            error
        );


        addAssistantMessage(
            `Error: ${escapeHTML(
                error.message
            )}`
        );


    } finally {

        sendBtn.disabled =
            false;

        questionInput.focus();
    }
}


// =====================================================
// USER MESSAGE
// =====================================================

function addUserMessage(text) {

    const element =
        document.createElement(
            "div"
        );


    element.className =
        "message user-message";


    element.innerHTML = `

        <div class="message-content">

            <p>
                ${escapeHTML(text)}
            </p>

        </div>

    `;


    chatMessages.appendChild(
        element
    );


    scrollChat();
}


// =====================================================
// ASSISTANT MESSAGE
// =====================================================

function addAssistantMessage(
    text,
    sources = [],
    cacheHit = false
) {

    const sourceHtml =
        sources.length

            ? `
                <div class="sources">

                    <div class="sources-title">

                        <i class="bi bi-book"></i>

                        Sources

                    </div>


                    ${sources
                        .map(
                            (source) => `

                                <div class="source-item">

                                    <i class="bi bi-file-earmark-text"></i>

                                    <span>

                                        ${escapeHTML(
                                            source.filename
                                        )}

                                        <small>

                                            ${
                                                source.page_number
                                                    ? `Page ${source.page_number}`
                                                    : "Document"
                                            }

                                        </small>

                                    </span>

                                </div>

                            `
                        )
                        .join("")}

                </div>
            `

            : "";


    const cacheBadge =
        cacheHit

            ? `
                <small class="cache-badge">
                    Semantic cache
                </small>
            `

            : "";


    const element =
        document.createElement(
            "div"
        );


    element.className =
        "message assistant-message";


    element.innerHTML = `

        <div class="message-avatar">

            <i class="bi bi-stars"></i>

        </div>


        <div class="message-content">

            <div class="assistant-answer">

                ${formatAnswer(text)}

            </div>


            ${cacheBadge}


            ${sourceHtml}

        </div>

    `;


    chatMessages.appendChild(
        element
    );


    scrollChat();
}


// =====================================================
// FORMAT ANSWER
// =====================================================

function formatAnswer(text) {

    let html =
        escapeHTML(text);


    // ---------------------------------------------
    // Bold
    // ---------------------------------------------

    html =
        html.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );


    // ---------------------------------------------
    // Bullet points
    // ---------------------------------------------

    html =
        html.replace(
            /^- (.*)$/gm,
            "• $1"
        );


    // ---------------------------------------------
    // New lines
    // ---------------------------------------------

    html =
        html.replace(
            /\n/g,
            "<br>"
        );


    return `<p>${html}</p>`;
}


// =====================================================
// SCROLL CHAT
// =====================================================

function scrollChat() {

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


// =====================================================
// FILE SIZE
// =====================================================

function formatBytes(bytes) {

    return `${(
        bytes / 1024 / 1024
    ).toFixed(2)} MB`;
}


// =====================================================
// HTML SECURITY
// =====================================================

function escapeHTML(text) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        text ?? "";


    return div.innerHTML;
}


// =====================================================
// INITIAL LOAD
// =====================================================

loadDocuments().catch(
    (error) => {

        console.error(
            "Initial document loading failed:",
            error
        );
    }
);