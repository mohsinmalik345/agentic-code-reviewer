"use strict";

const DASHBOARD_QUERY = `
  query Dashboard {
    health { status indexingMode modelId chatModelId runtime }
    repositories {
      id name description provider status gitlabUrl ref exactCommit knowledgeBaseUri
      graphNodeCount graphRelationshipCount lastScanRunId createdAt updatedAt indexedAt
    }
    indexingJobs(limit: 50) {
      id requestedRepositoryId status stage scanRunId errorCode canCancel
      createdAt updatedAt startedAt completedAt cancellationRequestedAt
    }
    graphSnapshot {
      available source generatedAt repositoryIds
      statistics {
        nodeCount relationshipCount
        nodesByType { key value }
        relationshipsByType { key value }
      }
    }
  }
`;

const JOBS_QUERY = `
  query LiveJobs {
    repositories {
      id name description provider status gitlabUrl ref exactCommit knowledgeBaseUri
      graphNodeCount graphRelationshipCount lastScanRunId createdAt updatedAt indexedAt
    }
    indexingJobs(limit: 50) {
      id requestedRepositoryId status stage scanRunId errorCode canCancel
      createdAt updatedAt startedAt completedAt cancellationRequestedAt
    }
    graphSnapshot {
      available source generatedAt repositoryIds
      statistics {
        nodeCount relationshipCount
        nodesByType { key value }
        relationshipsByType { key value }
      }
    }
  }
`;

const ONBOARD_MUTATION = `
  mutation OnboardRepositories($input: OnboardRepositoriesInput!) {
    onboardRepositories(input: $input) {
      repositories {
        id name description provider status gitlabUrl ref exactCommit knowledgeBaseUri
        graphNodeCount graphRelationshipCount lastScanRunId createdAt updatedAt indexedAt
      }
      job {
        id requestedRepositoryId status stage scanRunId errorCode canCancel
        createdAt updatedAt startedAt completedAt cancellationRequestedAt
      }
    }
  }
`;

const REINDEX_MUTATION = `
  mutation Reindex($input: ReindexInput!) {
    reindex(input: $input) {
      id requestedRepositoryId status stage scanRunId errorCode canCancel
      createdAt updatedAt startedAt completedAt cancellationRequestedAt
    }
  }
`;

const CANCEL_JOB_MUTATION = `
  mutation CancelIndexingJob($input: CancelIndexingJobInput!) {
    cancelIndexingJob(input: $input) {
      id requestedRepositoryId status stage scanRunId errorCode canCancel
      createdAt updatedAt startedAt completedAt cancellationRequestedAt
    }
  }
`;

const JOB_EVENTS_QUERY = `
  query IndexingJobEvents($jobId: ID!) {
    indexingJobEvents(jobId: $jobId, limit: 500) {
      id jobId level stage code message errorType createdAt
    }
  }
`;

const ARCHITECTURE_CHAT_MUTATION = `
  mutation AskArchitecture($input: ArchitectureChatInput!) {
    askArchitecture(input: $input) {
      answerMarkdown
      sources { id title repositoryId section evidenceKind }
      limitations
      invocation { modelId inputTokens outputTokens totalTokens latencyMs }
    }
  }
`;

const JIRA_CONNECTIONS_QUERY = `
  query JiraConnections {
    jiraConnections(includeDisabled: true) {
      id name edition baseUrl authType username credentialEnv enabled credentialConfigured
      createdAt updatedAt
    }
  }
`;

const JIRA_PROJECT_MAPPINGS_QUERY = `
  query JiraProjectMappings($repositoryId: ID) {
    jiraProjectMappings(repositoryId: $repositoryId) {
      id connectionId repositoryId jiraProjectKey acceptanceCriteriaFields issueKeyPattern
      createdAt updatedAt
    }
  }
`;

const SAVE_JIRA_CONNECTION_MUTATION = `
  mutation SaveJiraConnection($input: JiraConnectionInput!) {
    saveJiraConnection(input: $input) {
      id name edition baseUrl authType username credentialEnv enabled credentialConfigured
      createdAt updatedAt
    }
  }
`;

const TEST_JIRA_CONNECTION_MUTATION = `
  mutation TestJiraConnection($input: JiraConnectionTestInput!) {
    testJiraConnection(input: $input) {
      ok message serverTitle serverVersion authenticatedUser
    }
  }
`;

const DELETE_JIRA_CONNECTION_MUTATION = `
  mutation DeleteJiraConnection($input: DeleteJiraConnectionInput!) {
    deleteJiraConnection(input: $input)
  }
`;

const SAVE_JIRA_PROJECT_MAPPING_MUTATION = `
  mutation SaveJiraProjectMapping($input: JiraProjectMappingInput!) {
    saveJiraProjectMapping(input: $input) {
      id connectionId repositoryId jiraProjectKey acceptanceCriteriaFields issueKeyPattern
      createdAt updatedAt
    }
  }
`;

const DELETE_JIRA_PROJECT_MAPPING_MUTATION = `
  mutation DeleteJiraProjectMapping($input: DeleteJiraProjectMappingInput!) {
    deleteJiraProjectMapping(input: $input)
  }
`;

const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);
const STAGES = ["queued", "materializing", "scanning", "publishing", "complete"];
const MAX_ONBOARD_REPOSITORIES = 50;
const JOB_HISTORY_PAGE_SIZE = 5;
const JOB_HISTORY_MAX_ITEMS = 50;
const NUMBER_FORMAT = new Intl.NumberFormat();
const RELATIVE_TIME = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const state = {
  csrfToken: null,
  repositories: [],
  jobs: [],
  jobHistoryPage: 1,
  graph: null,
  health: null,
  pollingTimer: null,
  refreshPromise: null,
  knowledgeText: "",
  knowledgeSource: null,
  knowledgeMode: "human",
  knowledgeRequest: 0,
  chatHistory: [],
  chatBusy: false,
  jiraConnections: [],
  jiraMappings: [],
  jiraConnectionsLoaded: false,
  jiraMappingsLoaded: false,
  jiraConnectionsError: "",
  jiraMappingsError: "",
  jiraLoadPromise: null,
  jiraEditingConnectionId: null,
};

const elements = {};

class AuthenticationError extends Error {}

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  restoreSession();
});

function cacheElements() {
  const ids = [
    "login-view", "dashboard-view", "login-form", "api-key", "toggle-key", "login-error",
    "refresh-button", "logout-button", "system-status", "engine-model", "engine-mode",
    "open-onboard", "open-onboard-secondary", "open-central-knowledge", "reindex-all",
    "onboard-dialog", "onboard-form", "onboard-error", "onboard-repository-rows",
    "onboard-repository-template", "add-onboard-repository", "onboard-row-count", "knowledge-dialog", "knowledge-copy",
    "knowledge-content", "knowledge-title", "knowledge-kicker", "knowledge-meta", "toast-region",
    "metric-repositories", "metric-indexed", "metric-nodes", "metric-relationships", "metric-jobs",
    "metric-job-stage", "graph-source", "graph-core-count", "graph-types", "repository-list",
    "activity-title", "job-list", "jobs-preview", "polling-state", "knowledge-reader", "knowledge-evidence",
    "job-pagination", "job-page-summary", "job-page-previous", "job-page-indicator", "job-page-next",
    "job-log-dialog", "job-log-title", "job-log-meta", "job-log-content",
    "chat-form", "chat-question", "chat-send", "chat-error", "chat-scope", "chat-clear",
    "chat-messages", "chat-model", "refresh-jira", "new-jira-connection",
    "jira-connection-form", "jira-connection-form-title", "jira-connection-editing",
    "jira-username-field", "jira-connection-error", "jira-test-result",
    "cancel-jira-connection-edit", "jira-connection-list", "jira-mapping-form",
    "jira-mapping-connection", "jira-mapping-repository", "jira-mapping-error",
    "jira-mapping-list", "save-jira-mapping",
  ];
  ids.forEach((id) => { elements[id] = document.getElementById(id); });
}

function bindEvents() {
  elements["login-form"].addEventListener("submit", login);
  elements["toggle-key"].addEventListener("click", toggleApiKey);
  elements["logout-button"].addEventListener("click", logout);
  elements["refresh-button"].addEventListener("click", () => refreshDashboard(true));
  elements["job-page-previous"].addEventListener("click", () => changeJobHistoryPage(-1));
  elements["job-page-next"].addEventListener("click", () => changeJobHistoryPage(1));
  elements["open-onboard"].addEventListener("click", openOnboardDialog);
  elements["open-onboard-secondary"].addEventListener("click", openOnboardDialog);
  elements["open-central-knowledge"].addEventListener("click", () => openKnowledge("central"));
  elements["reindex-all"].addEventListener("click", () => reindexRepository(null));
  elements["onboard-form"].addEventListener("submit", onboardRepository);
  elements["add-onboard-repository"].addEventListener("click", () => addOnboardRepositoryRow({}, true));
  elements["knowledge-copy"].addEventListener("click", copyKnowledge);
  elements["knowledge-reader"].addEventListener("click", () => switchKnowledgeMode("human"));
  elements["knowledge-evidence"].addEventListener("click", () => switchKnowledgeMode("evidence"));
  elements["chat-form"].addEventListener("submit", askArchitecture);
  elements["chat-clear"].addEventListener("click", clearArchitectureChat);
  elements["chat-scope"].addEventListener("change", clearArchitectureChat);
  elements["refresh-jira"].addEventListener("click", () => refreshJiraConfiguration(true));
  elements["new-jira-connection"].addEventListener("click", () => resetJiraConnectionForm(true));
  elements["jira-connection-form"].addEventListener("submit", saveJiraConnection);
  elements["jira-connection-form"].elements.edition.addEventListener("change", updateJiraUsernameField);
  elements["jira-connection-form"].elements.authType.addEventListener("change", updateJiraUsernameField);
  elements["cancel-jira-connection-edit"].addEventListener("click", () => resetJiraConnectionForm(true));
  elements["jira-mapping-form"].addEventListener("submit", saveJiraProjectMapping);
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => document.getElementById(button.dataset.closeDialog).close());
  });
  document.querySelectorAll(".modal, .knowledge-modal").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });
  setupNavigationObserver();
}

async function restoreSession() {
  setAuthPending(true);
  try {
    const response = await fetch("/auth/session", { credentials: "same-origin", headers: { Accept: "application/json" } });
    if (!response.ok) {
      showLogin();
      return;
    }
    const payload = await response.json();
    if (!payload.authenticated || typeof payload.csrfToken !== "string") {
      showLogin();
      return;
    }
    state.csrfToken = payload.csrfToken;
    showDashboard();
  } catch (_error) {
    showLogin("The server could not be reached. Confirm the application is running and try again.");
  } finally {
    setAuthPending(false);
  }
}

async function login(event) {
  event.preventDefault();
  const apiKey = elements["api-key"].value;
  if (!apiKey) {
    setLoginError("Enter the configured API key.");
    elements["api-key"].focus();
    return;
  }
  setLoginError("");
  setAuthPending(true);
  try {
    const response = await fetch("/auth/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ apiKey }),
    });
    const payload = await readJson(response);
    if (!response.ok || typeof payload.csrfToken !== "string") {
      throw new Error(errorMessage(payload, "The API key was not accepted."));
    }
    state.csrfToken = payload.csrfToken;
    elements["api-key"].value = "";
    showDashboard();
  } catch (error) {
    setLoginError(messageFrom(error));
  } finally {
    setAuthPending(false);
  }
}

function showDashboard() {
  elements["login-view"].hidden = true;
  elements["dashboard-view"].hidden = false;
  document.title = "Operations | AI Code Intelligence";
  document.getElementById("main-content").focus({ preventScroll: true });
  renderLoadingState();
  refreshDashboard(false);
  startPolling();
}

function showLogin(error = "") {
  stopPolling();
  state.csrfToken = null;
  state.refreshPromise = null;
  state.jobHistoryPage = 1;
  state.chatHistory = [];
  state.chatBusy = false;
  state.knowledgeSource = null;
  state.jiraConnections = [];
  state.jiraMappings = [];
  state.jiraConnectionsLoaded = false;
  state.jiraMappingsLoaded = false;
  state.jiraConnectionsError = "";
  state.jiraMappingsError = "";
  state.jiraEditingConnectionId = null;
  elements["dashboard-view"].hidden = true;
  elements["login-view"].hidden = false;
  document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
  document.title = "Sign in | AI Code Intelligence";
  setLoginError(error);
  elements["api-key"].focus();
}

function setAuthPending(pending) {
  const button = elements["login-form"].querySelector("button[type='submit']");
  button.disabled = pending;
  button.querySelector("span").textContent = pending ? "Verifying session..." : "Open control plane";
}

function setLoginError(message) {
  elements["login-error"].textContent = message;
  elements["api-key"].setAttribute("aria-invalid", message ? "true" : "false");
}

function toggleApiKey() {
  const revealing = elements["api-key"].type === "password";
  elements["api-key"].type = revealing ? "text" : "password";
  elements["toggle-key"].textContent = revealing ? "Hide" : "Show";
  elements["toggle-key"].setAttribute("aria-label", `${revealing ? "Hide" : "Show"} API key`);
}

function openOnboardDialog() {
  elements["onboard-error"].textContent = "";
  if (!onboardRepositoryRows().length) addOnboardRepositoryRow();
  elements["onboard-dialog"].showModal();
  onboardRepositoryRows()[0].querySelector('[name="gitlabUrl"]').focus();
}

function onboardRepositoryRows() {
  return Array.from(elements["onboard-repository-rows"].querySelectorAll("[data-onboard-row]"));
}

function addOnboardRepositoryRow(values = {}, focus = false) {
  if (onboardRepositoryRows().length >= MAX_ONBOARD_REPOSITORIES) {
    elements["onboard-error"].textContent = `You can add at most ${MAX_ONBOARD_REPOSITORIES} repositories at once.`;
    return null;
  }
  const fragment = elements["onboard-repository-template"].content.cloneNode(true);
  const row = fragment.querySelector("[data-onboard-row]");
  Object.entries(values).forEach(([name, value]) => {
    const control = row.querySelector(`[name="${cssEscape(name)}"]`);
    if (control && typeof value === "string") control.value = value;
  });
  row.querySelector("[data-remove-onboard-row]").addEventListener("click", () => {
    const rows = onboardRepositoryRows();
    const index = rows.indexOf(row);
    row.remove();
    updateOnboardRepositoryRows();
    const remaining = onboardRepositoryRows();
    const target = remaining[Math.min(index, remaining.length - 1)];
    if (target) target.querySelector('[name="gitlabUrl"]').focus();
  });
  elements["onboard-repository-rows"].append(row);
  updateOnboardRepositoryRows();
  if (focus) row.querySelector('[name="gitlabUrl"]').focus();
  return row;
}

function updateOnboardRepositoryRows() {
  const rows = onboardRepositoryRows();
  rows.forEach((row, index) => {
    row.querySelector("[data-onboard-row-title]").textContent = `Repository ${index + 1}`;
    const remove = row.querySelector("[data-remove-onboard-row]");
    remove.disabled = rows.length === 1;
    remove.setAttribute("aria-label", `Remove repository ${index + 1}`);
  });
  elements["add-onboard-repository"].disabled = rows.length >= MAX_ONBOARD_REPOSITORIES;
  elements["onboard-row-count"].textContent = `${rows.length} of ${MAX_ONBOARD_REPOSITORIES} repositories added.`;
}

function resetOnboardRepositoryRows() {
  elements["onboard-repository-rows"].replaceChildren();
  addOnboardRepositoryRow();
}

async function onboardRepository(event) {
  event.preventDefault();
  const form = elements["onboard-form"];
  if (!form.reportValidity()) return;
  const submit = form.querySelector("button[type='submit']");
  const repositories = onboardRepositoryRows().map((row) => {
    const values = new FormData();
    row.querySelectorAll("input, textarea").forEach((control) => {
      values.set(control.name, control.value);
    });
    return compactObject({
      gitlabUrl: values.get("gitlabUrl"),
      repositoryId: values.get("repositoryId"),
      name: values.get("name"),
      description: values.get("description"),
      ref: values.get("ref"),
    });
  });
  elements["onboard-error"].textContent = "";
  setButtonPending(submit, true, repositories.length === 1 ? "Registering..." : "Registering batch...");
  try {
    const data = await graphqlRequest(ONBOARD_MUTATION, { input: { repositories } });
    const payload = data.onboardRepositories;
    payload.repositories.forEach(upsertRepository);
    upsertJob(payload.job);
    renderDashboard();
    resetOnboardRepositoryRows();
    elements["onboard-dialog"].close();
    const count = payload.repositories.length;
    toast(`${count} ${count === 1 ? "repository was" : "repositories were"} registered. One portfolio indexing job is queued.`);
    startPolling(true);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) elements["onboard-error"].textContent = messageFrom(error);
  } finally {
    setButtonPending(submit, false);
  }
}

async function reindexRepository(repositoryId) {
  const scope = repositoryId ? repositoryName(repositoryId) : "the full repository portfolio";
  if (!window.confirm(`Queue an index update for ${scope}? Unchanged commits will be reused.`)) return;
  const trigger = repositoryId
    ? document.querySelector(`[data-reindex-id="${cssEscape(repositoryId)}"]`)
    : elements["reindex-all"];
  if (trigger) setButtonPending(trigger, true, "Queueing...");
  try {
    const data = await graphqlRequest(REINDEX_MUTATION, { input: compactObject({ repositoryId }) });
    upsertJob(data.reindex);
    renderDashboard();
    toast(`Reindex queued for ${scope}.`);
    startPolling(true);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) toast(messageFrom(error), "error");
  } finally {
    if (trigger) setButtonPending(trigger, false);
  }
}

async function cancelIndexingJob(job, trigger) {
  const active = job.status === "running";
  const prompt = active
    ? "Request a safe stop? The current Bedrock invocation must return before the worker can stop."
    : "Cancel this queued indexing job?";
  if (!window.confirm(prompt)) return;
  setButtonPending(trigger, true, active ? "Requesting stop..." : "Cancelling...");
  try {
    const data = await graphqlRequest(CANCEL_JOB_MUTATION, { input: { jobId: job.id } });
    upsertJob(data.cancelIndexingJob);
    renderDashboard();
    toast(active ? "Safe cancellation requested." : "Queued job cancelled.");
    startPolling(true);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) toast(messageFrom(error), "error");
  } finally {
    setButtonPending(trigger, false);
  }
}

function startPolling(immediate = false) {
  stopPolling();
  if (!state.csrfToken || elements["dashboard-view"].hidden) return;
  state.pollingTimer = window.setTimeout(pollDashboard, immediate ? 500 : pollingDelay());
  elements["polling-state"].classList.remove("paused");
}

function stopPolling() {
  if (state.pollingTimer !== null) window.clearTimeout(state.pollingTimer);
  state.pollingTimer = null;
}

async function logout() {
  stopPolling();
  try {
    await fetch("/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: state.csrfToken ? { "x-csrf-token": state.csrfToken } : {},
    });
  } finally {
    showLogin();
  }
}

async function pollDashboard() {
  if (!state.csrfToken || document.hidden) {
    startPolling();
    return;
  }
  try {
    const data = await graphqlRequest(JOBS_QUERY);
    applyDashboardData(data);
    renderMetrics();
    renderGraph();
    renderRepositories();
    renderJobs();
    renderChatScope();
    setSystemStatus("online", "Systems operational");
  } catch (error) {
    if (!(error instanceof AuthenticationError)) setSystemStatus("offline", "Polling delayed");
  } finally {
    if (state.csrfToken) startPolling();
  }
}

function pollingDelay() {
  return state.jobs.some((job) => ACTIVE_JOB_STATUSES.has(job.status)) ? 3000 : 12000;
}

async function refreshDashboard(announce = false) {
  if (state.refreshPromise) return state.refreshPromise;
  elements["refresh-button"].disabled = true;
  elements["refresh-button"].textContent = "Refreshing...";
  setSystemStatus("connecting", "Synchronizing");
  refreshJiraConfiguration(false);
  state.refreshPromise = (async () => {
    try {
      const data = await graphqlRequest(DASHBOARD_QUERY);
      applyDashboardData(data);
      renderDashboard();
      setSystemStatus("online", "Systems operational");
      if (announce) toast("Dashboard refreshed.");
    } catch (error) {
      if (!(error instanceof AuthenticationError)) {
        setSystemStatus("offline", "Data unavailable");
        renderDashboardError(messageFrom(error));
        if (announce) toast(messageFrom(error), "error");
      }
    } finally {
      elements["refresh-button"].disabled = false;
      elements["refresh-button"].textContent = "Refresh";
      state.refreshPromise = null;
    }
  })();
  return state.refreshPromise;
}

function applyDashboardData(data) {
  if (data.health) state.health = data.health;
  if (Array.isArray(data.repositories)) state.repositories = data.repositories;
  if (Array.isArray(data.indexingJobs)) state.jobs = data.indexingJobs;
  if (data.graphSnapshot) state.graph = data.graphSnapshot;
}

function setSystemStatus(status, text) {
  elements["system-status"].classList.remove("online", "offline");
  if (status === "online" || status === "offline") elements["system-status"].classList.add(status);
  elements["system-status"].querySelector("span:last-child").textContent = text;
}

function renderDashboard() {
  renderHealth();
  renderMetrics();
  renderGraph();
  renderRepositories();
  renderJobs();
  renderChatScope();
  renderJiraSelectors();
}

function renderLoadingState() {
  elements["repository-list"].setAttribute("aria-busy", "true");
  elements["job-list"].setAttribute("aria-busy", "true");
  ["metric-repositories", "metric-nodes", "metric-relationships", "metric-jobs"].forEach((id) => {
    elements[id].textContent = "—";
  });
  if (!state.jiraConnectionsLoaded) {
    elements["jira-connection-list"].setAttribute("aria-busy", "true");
    elements["jira-connection-list"].replaceChildren(emptyState("Loading Jira connections", "Checking server-side integration settings.", false));
  }
  if (!state.jiraMappingsLoaded) {
    elements["jira-mapping-list"].setAttribute("aria-busy", "true");
    elements["jira-mapping-list"].replaceChildren(emptyState("Loading repository mappings", "Checking which Jira projects are linked.", false));
  }
  renderJiraSelectors();
}

function renderDashboardError(message) {
  if (!state.repositories.length) {
    elements["repository-list"].replaceChildren(emptyState("Repository data unavailable", message, true));
  }
  if (!state.jobs.length) {
    elements["job-list"].replaceChildren(emptyState("Activity data unavailable", message, true));
    elements["jobs-preview"].replaceChildren(emptyState("Activity unavailable", message, true));
  }
}

async function refreshJiraConfiguration(announce = false) {
  if (!state.csrfToken) return;
  if (state.jiraLoadPromise) return state.jiraLoadPromise;
  const trigger = elements["refresh-jira"];
  if (announce) setButtonPending(trigger, true, "Refreshing...");
  state.jiraConnectionsError = "";
  state.jiraMappingsError = "";
  if (!state.jiraConnectionsLoaded || !state.jiraMappingsLoaded) renderLoadingState();

  state.jiraLoadPromise = (async () => {
    const [connectionsResult, mappingsResult] = await Promise.allSettled([
      graphqlRequest(JIRA_CONNECTIONS_QUERY),
      graphqlRequest(JIRA_PROJECT_MAPPINGS_QUERY, { repositoryId: null }),
    ]);
    const results = [connectionsResult, mappingsResult];
    if (results.some((result) => result.status === "rejected" && result.reason instanceof AuthenticationError)) return;

    if (connectionsResult.status === "fulfilled") {
      state.jiraConnections = Array.isArray(connectionsResult.value.jiraConnections)
        ? connectionsResult.value.jiraConnections
        : [];
      state.jiraConnectionsLoaded = true;
    } else {
      state.jiraConnectionsLoaded = true;
      state.jiraConnectionsError = messageFrom(connectionsResult.reason);
    }

    if (mappingsResult.status === "fulfilled") {
      state.jiraMappings = Array.isArray(mappingsResult.value.jiraProjectMappings)
        ? mappingsResult.value.jiraProjectMappings
        : [];
      state.jiraMappingsLoaded = true;
    } else {
      state.jiraMappingsLoaded = true;
      state.jiraMappingsError = messageFrom(mappingsResult.reason);
    }

    renderJiraConfiguration();
    const errors = [state.jiraConnectionsError, state.jiraMappingsError].filter(Boolean);
    if (announce) {
      if (errors.length) toast(errors.join(" · "), "error");
      else toast("Jira configuration refreshed.");
    }
  })();

  try {
    await state.jiraLoadPromise;
  } finally {
    state.jiraLoadPromise = null;
    if (announce) setButtonPending(trigger, false);
  }
}

function renderJiraConfiguration() {
  renderJiraConnections();
  renderJiraMappings();
  renderJiraSelectors();
}

function renderJiraConnections() {
  const container = elements["jira-connection-list"];
  container.setAttribute("aria-busy", "false");
  container.replaceChildren();
  if (state.jiraConnectionsError) {
    container.append(jiraErrorState("Unable to load Jira connections", state.jiraConnectionsError));
    return;
  }
  if (!state.jiraConnections.length) {
    const empty = emptyState(
      "No Jira connections yet",
      "Add a Jira Cloud or Data Center instance above. Only an environment-variable name is saved.",
      false,
    );
    const add = element("button", "button button-secondary", "Add first connection");
    add.type = "button";
    add.addEventListener("click", () => resetJiraConnectionForm(true));
    empty.append(add);
    container.append(empty);
    return;
  }
  [...state.jiraConnections]
    .sort((a, b) => String(a.name).localeCompare(String(b.name)))
    .forEach((connection) => container.append(jiraConnectionCard(connection)));
}

function jiraConnectionCard(connection) {
  const card = element("article", "jira-card");
  const header = element("div", "jira-card-header");
  const title = element("div", "jira-card-title");
  title.append(element("h4", "", connection.name), element("code", "", connection.id));
  header.append(title, statusBadge(connection.enabled ? "enabled" : "disabled"));

  const facts = element("dl", "jira-card-facts");
  appendFact(facts, "Edition", titleCase(connection.edition));
  appendFact(facts, "Authentication", titleCase(connection.authType));
  appendFact(facts, "Credential", connection.credentialConfigured ? "Environment ready" : "Not configured");
  appendFact(facts, "Environment", connection.credentialEnv || "Not set");
  if (connection.username) appendFact(facts, "Cloud user", connection.username);

  const actions = element("div", "jira-card-actions");
  const edit = element("button", "button button-quiet", "Edit");
  edit.type = "button";
  edit.addEventListener("click", () => editJiraConnection(connection));
  const test = element("button", "button button-secondary", "Test");
  test.type = "button";
  test.addEventListener("click", () => testJiraConnection(connection, test));
  const remove = element("button", "button button-danger", "Delete");
  remove.type = "button";
  remove.addEventListener("click", () => deleteJiraConnection(connection, remove));
  actions.append(edit, test, remove);

  card.append(header, element("p", "jira-card-description", connection.baseUrl), facts, actions);
  return card;
}

function editJiraConnection(connection) {
  const form = elements["jira-connection-form"];
  state.jiraEditingConnectionId = connection.id;
  form.elements.id.value = connection.id || "";
  form.elements.id.readOnly = true;
  form.elements.name.value = connection.name || "";
  form.elements.edition.value = connection.edition || "cloud";
  form.elements.baseUrl.value = connection.baseUrl || "";
  form.elements.authType.value = connection.authType || "api_token";
  form.elements.username.value = connection.username || "";
  form.elements.credentialEnv.value = connection.credentialEnv || "";
  form.elements.enabled.checked = Boolean(connection.enabled);
  elements["jira-connection-form-title"].textContent = `Edit ${connection.name}`;
  elements["jira-connection-editing"].hidden = false;
  elements["cancel-jira-connection-edit"].hidden = false;
  elements["jira-connection-error"].textContent = "";
  hideJiraTestResult();
  updateJiraUsernameField();
  form.scrollIntoView({ behavior: "smooth", block: "center" });
  form.elements.name.focus({ preventScroll: true });
}

function resetJiraConnectionForm(focus = false) {
  const form = elements["jira-connection-form"];
  form.reset();
  form.elements.id.readOnly = false;
  state.jiraEditingConnectionId = null;
  elements["jira-connection-form-title"].textContent = "Add a Jira connection";
  elements["jira-connection-editing"].hidden = true;
  elements["cancel-jira-connection-edit"].hidden = true;
  elements["jira-connection-error"].textContent = "";
  hideJiraTestResult();
  updateJiraUsernameField();
  if (focus) {
    form.scrollIntoView({ behavior: "smooth", block: "center" });
    form.elements.id.focus({ preventScroll: true });
  }
}

function updateJiraUsernameField() {
  const form = elements["jira-connection-form"];
  const cloud = form.elements.edition.value === "cloud";
  const expectedAuthType = cloud ? "api_token" : "personal_access_token";
  const previousAuthType = form.elements.authType.value;
  form.elements.authType.value = expectedAuthType;
  form.elements.authType.disabled = true;
  const credential = form.elements.credentialEnv;
  if (!credential.value || credential.value === "JIRA_CLOUD_API_TOKEN" || credential.value === "JIRA_DATA_CENTER_PAT") {
    credential.value = cloud ? "JIRA_CLOUD_API_TOKEN" : "JIRA_DATA_CENTER_PAT";
  }
  const required = cloud;
  elements["jira-username-field"].hidden = !required;
  form.elements.username.required = required;
  if (!cloud && previousAuthType !== expectedAuthType) form.elements.username.value = "";
}

async function saveJiraConnection(event) {
  event.preventDefault();
  const form = elements["jira-connection-form"];
  updateJiraUsernameField();
  if (!form.reportValidity()) return;
  const submit = form.querySelector("button[type='submit']");
  const values = new FormData(form);
  const usesCloudUsername = values.get("edition") === "cloud";
  const input = compactObject({
    id: values.get("id"),
    name: values.get("name"),
    edition: values.get("edition"),
    baseUrl: values.get("baseUrl"),
    authType: form.elements.authType.value,
    username: usesCloudUsername ? values.get("username") : "",
    credentialEnv: values.get("credentialEnv"),
    enabled: form.elements.enabled.checked,
  });
  elements["jira-connection-error"].textContent = "";
  hideJiraTestResult();
  setButtonPending(submit, true, "Saving...");
  try {
    const data = await graphqlRequest(SAVE_JIRA_CONNECTION_MUTATION, { input });
    const connection = data.saveJiraConnection;
    upsertJiraConnection(connection);
    state.jiraConnectionsLoaded = true;
    state.jiraConnectionsError = "";
    renderJiraConfiguration();
    resetJiraConnectionForm(false);
    toast(`${connection.name} was saved. Configure ${connection.credentialEnv} in the server environment before testing.`);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) elements["jira-connection-error"].textContent = messageFrom(error);
  } finally {
    setButtonPending(submit, false);
  }
}

async function testJiraConnection(connection, trigger) {
  elements["jira-connection-error"].textContent = "";
  hideJiraTestResult();
  setButtonPending(trigger, true, "Testing...");
  try {
    const data = await graphqlRequest(TEST_JIRA_CONNECTION_MUTATION, { input: { connectionId: connection.id } });
    const result = data.testJiraConnection;
    showJiraTestResult(connection, result);
    if (result.ok) toast(`${connection.name} connected successfully.`);
    else toast(result.message || `${connection.name} could not connect.`, "error");
  } catch (error) {
    if (!(error instanceof AuthenticationError)) {
      const message = messageFrom(error);
      showJiraTestResult(connection, { ok: false, message });
      toast(message, "error");
    }
  } finally {
    setButtonPending(trigger, false);
  }
}

function showJiraTestResult(connection, result) {
  const container = elements["jira-test-result"];
  container.replaceChildren();
  container.hidden = false;
  container.classList.toggle("error", !result.ok);
  const title = result.ok
    ? `Connected${result.serverTitle ? ` to ${result.serverTitle}` : ""}`
    : `Connection test failed for ${connection.name}`;
  container.append(element("strong", "", title), element("p", "", result.message || "No details were returned."));
  const details = [];
  if (result.serverVersion) details.push(`Server ${result.serverVersion}`);
  if (result.authenticatedUser) details.push(`Authenticated as ${result.authenticatedUser}`);
  if (details.length) container.append(element("small", "", details.join(" · ")));
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideJiraTestResult() {
  const container = elements["jira-test-result"];
  container.hidden = true;
  container.classList.remove("error");
  container.replaceChildren();
}

async function deleteJiraConnection(connection, trigger) {
  const mappedCount = state.jiraMappings.filter((mapping) => mapping.connectionId === connection.id).length;
  const mappingWarning = mappedCount
    ? ` This will also remove ${mappedCount} repository ${mappedCount === 1 ? "mapping" : "mappings"}.`
    : "";
  if (!window.confirm(`Delete the Jira connection “${connection.name}”?${mappingWarning}`)) return;
  setButtonPending(trigger, true, "Deleting...");
  try {
    const data = await graphqlRequest(DELETE_JIRA_CONNECTION_MUTATION, { input: { connectionId: connection.id } });
    if (data.deleteJiraConnection !== true) throw new Error("The Jira connection was already removed.");
    state.jiraConnections = state.jiraConnections.filter((item) => item.id !== connection.id);
    state.jiraMappings = state.jiraMappings.filter((mapping) => mapping.connectionId !== connection.id);
    if (state.jiraEditingConnectionId === connection.id) resetJiraConnectionForm(false);
    renderJiraConfiguration();
    toast(`${connection.name} was deleted.`);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) toast(messageFrom(error), "error");
  } finally {
    setButtonPending(trigger, false);
  }
}

function renderJiraSelectors() {
  const connectionSelect = elements["jira-mapping-connection"];
  const repositorySelect = elements["jira-mapping-repository"];
  const selectedConnection = connectionSelect.value;
  const selectedRepository = repositorySelect.value;

  connectionSelect.replaceChildren();
  connectionSelect.append(new Option(
    state.jiraConnections.length ? "Select a Jira connection" : "Add a Jira connection first",
    "",
  ));
  [...state.jiraConnections]
    .filter((connection) => connection.enabled)
    .sort((a, b) => String(a.name).localeCompare(String(b.name)))
    .forEach((connection) => {
      const suffix = connection.enabled ? "" : " (disabled)";
      connectionSelect.append(new Option(`${connection.name}${suffix}`, connection.id));
    });
  if (state.jiraConnections.some((connection) => connection.id === selectedConnection)) {
    connectionSelect.value = selectedConnection;
  }

  repositorySelect.replaceChildren();
  repositorySelect.append(new Option(
    state.repositories.length ? "Select a repository" : "Add a repository first",
    "",
  ));
  [...state.repositories]
    .sort((a, b) => String(a.name).localeCompare(String(b.name)))
    .forEach((repository) => repositorySelect.append(new Option(repository.name, repository.id)));
  if (state.repositories.some((repository) => repository.id === selectedRepository)) {
    repositorySelect.value = selectedRepository;
  }

  elements["save-jira-mapping"].disabled = !state.jiraConnections.some((connection) => connection.enabled)
    || !state.repositories.length;
}

function renderJiraMappings() {
  const container = elements["jira-mapping-list"];
  container.setAttribute("aria-busy", "false");
  container.replaceChildren();
  if (state.jiraMappingsError) {
    container.append(jiraErrorState("Unable to load Jira mappings", state.jiraMappingsError));
    return;
  }
  if (!state.jiraMappings.length) {
    container.append(emptyState(
      "No repository mappings yet",
      "Choose a Jira connection, repository, and Jira project key above to create one.",
      false,
    ));
    return;
  }
  [...state.jiraMappings]
    .sort((a, b) => String(a.repositoryId).localeCompare(String(b.repositoryId)))
    .forEach((mapping) => container.append(jiraMappingCard(mapping)));
}

function jiraMappingCard(mapping) {
  const connection = state.jiraConnections.find((item) => item.id === mapping.connectionId);
  const repository = state.repositories.find((item) => item.id === mapping.repositoryId);
  const card = element("article", "jira-card");
  const header = element("div", "jira-card-header");
  const title = element("div", "jira-card-title");
  title.append(
    element("h4", "", `${mapping.jiraProjectKey} · ${repository ? repository.name : mapping.repositoryId}`),
    element("code", "", mapping.id),
  );
  header.append(title, element("span", "chip", "Mapped"));

  const facts = element("dl", "jira-card-facts");
  appendFact(facts, "Connection", connection ? connection.name : mapping.connectionId);
  appendFact(facts, "Repository", repository ? repository.name : mapping.repositoryId);
  appendFact(facts, "Project key", mapping.jiraProjectKey);
  const patternFact = element("div");
  patternFact.append(element("dt", "", "Issue-key pattern"), element("dd"));
  patternFact.querySelector("dd").append(element("code", "", mapping.issueKeyPattern));
  facts.append(patternFact);

  const tags = element("div", "jira-field-tags");
  const fields = Array.isArray(mapping.acceptanceCriteriaFields) ? mapping.acceptanceCriteriaFields : [];
  if (fields.length) fields.forEach((field) => tags.append(element("span", "jira-field-tag", field)));
  else tags.append(element("span", "jira-field-tag", "No acceptance-criteria fields"));

  const actions = element("div", "jira-card-actions");
  const remove = element("button", "button button-danger", "Delete mapping");
  remove.type = "button";
  remove.addEventListener("click", () => deleteJiraProjectMapping(mapping, remove));
  actions.append(remove);
  card.append(header, facts, tags, actions);
  return card;
}

async function saveJiraProjectMapping(event) {
  event.preventDefault();
  const form = elements["jira-mapping-form"];
  const pattern = form.elements.issueKeyPattern.value.trim();
  if (!form.reportValidity()) return;
  const values = new FormData(form);
  const acceptanceCriteriaFields = [...new Set(
    String(values.get("acceptanceCriteriaFields") || "")
      .split(",")
      .map((field) => field.trim())
      .filter(Boolean),
  )];
  const input = {
    connectionId: String(values.get("connectionId")).trim(),
    repositoryId: String(values.get("repositoryId")).trim(),
    jiraProjectKey: String(values.get("jiraProjectKey")).trim().toUpperCase(),
    acceptanceCriteriaFields,
    issueKeyPattern: pattern,
  };
  const submit = form.querySelector("button[type='submit']");
  elements["jira-mapping-error"].textContent = "";
  setButtonPending(submit, true, "Saving...");
  try {
    const data = await graphqlRequest(SAVE_JIRA_PROJECT_MAPPING_MUTATION, { input });
    const mapping = data.saveJiraProjectMapping;
    upsertJiraMapping(mapping);
    state.jiraMappingsLoaded = true;
    state.jiraMappingsError = "";
    renderJiraMappings();
    form.reset();
    renderJiraSelectors();
    toast(`${repositoryName(mapping.repositoryId)} is mapped to Jira project ${mapping.jiraProjectKey}.`);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) elements["jira-mapping-error"].textContent = messageFrom(error);
  } finally {
    setButtonPending(submit, false);
  }
}

async function deleteJiraProjectMapping(mapping, trigger) {
  const repository = repositoryName(mapping.repositoryId);
  if (!window.confirm(`Delete the ${repository} → ${mapping.jiraProjectKey} Jira mapping?`)) return;
  setButtonPending(trigger, true, "Deleting...");
  try {
    const data = await graphqlRequest(DELETE_JIRA_PROJECT_MAPPING_MUTATION, { input: { mappingId: mapping.id } });
    if (data.deleteJiraProjectMapping !== true) throw new Error("The Jira mapping was already removed.");
    state.jiraMappings = state.jiraMappings.filter((item) => item.id !== mapping.id);
    renderJiraMappings();
    toast(`The Jira mapping for ${repository} was deleted.`);
  } catch (error) {
    if (!(error instanceof AuthenticationError)) toast(messageFrom(error), "error");
  } finally {
    setButtonPending(trigger, false);
  }
}

function jiraErrorState(title, message) {
  const empty = emptyState(title, message, true);
  const retry = element("button", "button button-secondary", "Try again");
  retry.type = "button";
  retry.addEventListener("click", () => refreshJiraConfiguration(true));
  empty.append(retry);
  return empty;
}

function upsertJiraConnection(connection) {
  state.jiraConnections = [connection, ...state.jiraConnections.filter((item) => item.id !== connection.id)];
}

function upsertJiraMapping(mapping) {
  state.jiraMappings = [mapping, ...state.jiraMappings.filter((item) => item.id !== mapping.id)];
}

function renderHealth() {
  if (!state.health) return;
  elements["engine-model"].textContent = state.health.modelId || "Configurable model";
  elements["engine-mode"].textContent = `${state.health.indexingMode || "unknown"} · ${state.health.runtime || "runtime unknown"}`;
  elements["chat-model"].textContent = state.health.chatModelId
    ? `Model: ${state.health.chatModelId} · grounded in reader editions + Neo4j`
    : "Grounded Bedrock model";
}

function renderMetrics() {
  const indexed = state.repositories.filter((repository) => repository.status === "indexed").length;
  const active = state.jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status));
  const statistics = state.graph && state.graph.statistics;
  elements["metric-repositories"].textContent = NUMBER_FORMAT.format(state.repositories.length);
  elements["metric-indexed"].textContent = `${indexed} indexed · ${state.repositories.length - indexed} pending`;
  elements["metric-nodes"].textContent = statistics ? compactNumber(statistics.nodeCount) : "0";
  elements["metric-relationships"].textContent = statistics ? compactNumber(statistics.relationshipCount) : "0";
  elements["metric-jobs"].textContent = NUMBER_FORMAT.format(active.length);
  elements["metric-job-stage"].textContent = active[0] ? titleCase(active[0].stage) : "Queue is clear";
}

function renderGraph() {
  const graph = state.graph;
  const statistics = graph && graph.statistics;
  elements["graph-source"].textContent = graph && graph.available ? `${graph.source} · ${formatRelative(graph.generatedAt)}` : "No snapshot";
  elements["graph-core-count"].textContent = statistics ? compactNumber(statistics.nodeCount) : "0";
  const types = statistics && Array.isArray(statistics.nodesByType)
    ? [...statistics.nodesByType].sort((left, right) => right.value - left.value).slice(0, 4)
    : [];
  elements["graph-types"].replaceChildren();
  if (!types.length) {
    elements["graph-types"].append(element("p", "graph-empty-note", "Run the first index to populate topology."));
    return;
  }
  const maximum = Math.max(...types.map((item) => item.value), 1);
  types.forEach((item) => {
    const bar = element("div", "type-bar");
    const track = element("progress");
    track.max = maximum;
    track.value = item.value;
    track.setAttribute("aria-label", `${titleCase(item.key)}: ${NUMBER_FORMAT.format(item.value)}`);
    const label = element("small");
    label.append(element("span", "", titleCase(item.key)), element("span", "", compactNumber(item.value)));
    bar.append(track, label);
    elements["graph-types"].append(bar);
  });
}

function renderRepositories() {
  const container = elements["repository-list"];
  container.setAttribute("aria-busy", "false");
  container.replaceChildren();
  if (!state.repositories.length) {
    const empty = emptyState(
      "No repositories registered",
      "Add a GitLab clone URL to create the first graph and knowledge base.",
      false,
    );
    const add = element("button", "button button-primary", "Add first repository");
    add.type = "button";
    add.addEventListener("click", openOnboardDialog);
    empty.append(add);
    container.append(empty);
    return;
  }
  [...state.repositories]
    .sort((left, right) => left.name.localeCompare(right.name))
    .forEach((repository) => container.append(repositoryCard(repository)));
}

function renderChatScope() {
  const select = elements["chat-scope"];
  const selected = select.value;
  const options = [element("option", "", "All repositories + central")];
  options[0].value = "";
  state.repositories
    .filter((repository) => repository.status === "indexed" && repository.knowledgeBaseUri)
    .sort((left, right) => left.name.localeCompare(right.name))
    .forEach((repository) => {
      const option = element("option", "", `${repository.name} + central`);
      option.value = repository.id;
      options.push(option);
    });
  select.replaceChildren(...options);
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
}

function repositoryCard(repository) {
  const card = element("article", "repository-card");
  const header = element("header", "repository-card-header");
  const identity = element("div", "repository-identity");
  const glyph = element("span", "repository-glyph", initials(repository.name));
  glyph.setAttribute("aria-hidden", "true");
  const nameBlock = element("div");
  nameBlock.append(element("h3", "", repository.name), element("code", "repository-id", repository.id));
  identity.append(glyph, nameBlock);
  header.append(identity, statusBadge(repository.status));

  const description = element(
    "p",
    "repository-description",
    repository.description || "No repository description has been provided.",
  );
  const facts = element("dl", "repository-facts");
  appendFact(facts, "Provider", titleCase(repository.provider));
  appendFact(facts, "Ref", repository.ref || "configured source");
  appendFact(facts, "Indexed", repository.indexedAt ? formatRelative(repository.indexedAt) : "not yet");
  const stats = element("div", "repository-stats");
  stats.append(statBlock(repository.graphNodeCount, "nodes"), statBlock(repository.graphRelationshipCount, "relations"));

  const actions = element("footer", "repository-actions");
  const view = element("button", "button button-secondary", "View knowledge");
  view.type = "button";
  view.disabled = !repository.knowledgeBaseUri;
  view.title = repository.knowledgeBaseUri ? "Open generated repository knowledge" : "Available after a successful index";
  view.addEventListener("click", () => openKnowledge("repository", repository));
  const reindex = element("button", "button button-quiet", "Reindex");
  reindex.type = "button";
  reindex.dataset.reindexId = repository.id;
  reindex.addEventListener("click", () => reindexRepository(repository.id));
  actions.append(view, reindex);
  card.append(header, description, facts, stats, actions);
  return card;
}

function appendFact(list, term, description) {
  const wrapper = element("div");
  wrapper.append(element("dt", "", term), element("dd", "", description));
  list.append(wrapper);
}

function statBlock(value, label) {
  const block = element("span");
  block.append(element("strong", "", compactNumber(value || 0)), document.createTextNode(` ${label}`));
  return block;
}

function statusBadge(status) {
  const badge = element("span", `repository-status ${status}`);
  badge.append(element("i"), document.createTextNode(titleCase(status)));
  return badge;
}

function renderJobs() {
  const jobs = orderedRecentJobs(state.jobs);
  elements["job-list"].setAttribute("aria-busy", "false");
  elements["job-list"].replaceChildren();
  elements["jobs-preview"].replaceChildren();
  if (!jobs.length) {
    state.jobHistoryPage = 1;
    elements["job-pagination"].hidden = true;
    elements["job-list"].append(emptyState("No indexing jobs yet", "Onboard or reindex a repository to start the worker.", false));
    elements["jobs-preview"].append(emptyState("Queue is empty", "No jobs have been submitted.", false));
    return;
  }
  const pageCount = Math.ceil(jobs.length / JOB_HISTORY_PAGE_SIZE);
  state.jobHistoryPage = Math.min(Math.max(state.jobHistoryPage, 1), pageCount);
  const pageStart = (state.jobHistoryPage - 1) * JOB_HISTORY_PAGE_SIZE;
  const pageJobs = jobs.slice(pageStart, pageStart + JOB_HISTORY_PAGE_SIZE);
  pageJobs.forEach((job) => elements["job-list"].append(jobRow(job, false)));
  jobs.slice(0, 4).forEach((job) => elements["jobs-preview"].append(jobRow(job, true)));
  renderJobPagination(jobs.length, pageStart, pageJobs.length, pageCount);
}

function orderedRecentJobs(jobs) {
  return [...jobs]
    .sort((left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt))
    .slice(0, JOB_HISTORY_MAX_ITEMS);
}

function changeJobHistoryPage(offset) {
  const pageCount = Math.max(1, Math.ceil(orderedRecentJobs(state.jobs).length / JOB_HISTORY_PAGE_SIZE));
  const nextPage = Math.min(Math.max(state.jobHistoryPage + offset, 1), pageCount);
  if (nextPage === state.jobHistoryPage) return;
  state.jobHistoryPage = nextPage;
  renderJobs();
  elements["activity-title"].scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderJobPagination(total, pageStart, pageLength, pageCount) {
  const first = pageStart + 1;
  const last = pageStart + pageLength;
  const cappedLabel = total === JOB_HISTORY_MAX_ITEMS ? "latest " : "";
  elements["job-page-summary"].textContent = `Showing ${first}\u2013${last} of the ${cappedLabel}${total} jobs available in this view`;
  elements["job-page-indicator"].textContent = `Page ${state.jobHistoryPage} of ${pageCount}`;
  elements["job-page-previous"].disabled = state.jobHistoryPage === 1;
  elements["job-page-next"].disabled = state.jobHistoryPage === pageCount;
  elements["job-pagination"].hidden = false;
}

function jobRow(job, compact) {
  const row = element("article", `job-row${compact ? " compact" : ""}`);
  const main = element("div", "job-main");
  const title = job.requestedRepositoryId ? repositoryName(job.requestedRepositoryId) : "Full portfolio";
  main.append(element("h3", "", title));
  const summary = job.errorCode
    ? `${titleCase(job.errorCode)} · ${formatRelative(job.updatedAt)}`
    : job.cancellationRequestedAt && job.status === "running"
      ? `Stop requested · ${formatRelative(job.cancellationRequestedAt)}`
    : `${titleCase(job.stage)} · ${formatRelative(job.updatedAt)}`;
  main.append(element("p", "", summary));
  if (!compact) {
    main.append(jobProgress(job));
    const actions = element("div", "job-actions");
    const logs = element(
      "button",
      `button ${job.status === "failed" ? "button-danger" : "button-quiet"}`,
      job.status === "failed" ? "View failure log" : "View event log",
    );
    logs.type = "button";
    logs.addEventListener("click", () => openJobLog(job));
    actions.append(logs);
    if (job.canCancel) {
      const cancel = element(
        "button",
        "button button-danger",
        job.status === "queued" ? "Cancel job" : "Stop safely",
      );
      cancel.type = "button";
      cancel.addEventListener("click", () => cancelIndexingJob(job, cancel));
      actions.append(cancel);
    } else if (job.cancellationRequestedAt && job.status === "running") {
      const pending = element("button", "button button-quiet", "Stop requested");
      pending.type = "button";
      pending.disabled = true;
      actions.append(pending);
    }
    main.append(actions);
  }
  row.append(main, element("span", `job-status ${job.status}`, titleCase(job.status)));
  return row;
}

function jobProgress(job) {
  const list = element("ol", "job-progress");
  list.setAttribute("aria-label", `Indexing progress: ${titleCase(job.stage)}`);
  const current = Math.max(STAGES.indexOf(job.stage), 0);
  STAGES.forEach((stage, index) => {
    const item = element("li", index < current || job.status === "succeeded" ? "done" : index === current ? "current" : "");
    item.append(element("span", "", String(index + 1)), document.createTextNode(titleCase(stage)));
    list.append(item);
  });
  return list;
}

async function openJobLog(job) {
  elements["job-log-title"].textContent = `${job.requestedRepositoryId ? repositoryName(job.requestedRepositoryId) : "Full portfolio"} job log`;
  elements["job-log-meta"].textContent = `${titleCase(job.status)} · ${titleCase(job.stage)} · job ${shortId(job.id)}`;
  elements["job-log-content"].replaceChildren(loadingDocument());
  if (!elements["job-log-dialog"].open) elements["job-log-dialog"].showModal();
  try {
    const data = await graphqlRequest(JOB_EVENTS_QUERY, { jobId: job.id });
    const events = Array.isArray(data.indexingJobEvents) ? data.indexingJobEvents : [];
    elements["job-log-content"].replaceChildren();
    if (!events.length) {
      elements["job-log-content"].append(emptyState("No durable events", "This job predates event logging or has not started yet.", false));
      return;
    }
    events.forEach((event) => {
      const item = element("article", `job-event ${event.level || "info"}`);
      const marker = element("span", "job-event-marker", event.level === "error" ? "!" : event.level === "warning" ? "△" : "·");
      marker.setAttribute("aria-hidden", "true");
      const body = element("div", "job-event-body");
      const header = element("div", "job-event-header");
      header.append(
        element("strong", "", titleCase(event.code || event.stage)),
        element("time", "", formatDate(event.createdAt)),
      );
      const details = [titleCase(event.stage)];
      if (event.errorType) details.push(event.errorType);
      body.append(header, element("p", "", event.message), element("small", "", details.join(" · ")));
      item.append(marker, body);
      elements["job-log-content"].append(item);
    });
  } catch (error) {
    if (!(error instanceof AuthenticationError)) {
      elements["job-log-content"].replaceChildren(emptyState("Unable to load job events", messageFrom(error), true));
    }
  }
}

async function graphqlRequest(query, variables = {}) {
  if (!state.csrfToken) throw new AuthenticationError("No authenticated session.");
  let response;
  try {
    response = await fetch("/graphql", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "x-csrf-token": state.csrfToken,
      },
      body: JSON.stringify({ query, variables }),
    });
  } catch (_error) {
    throw new Error("The API could not be reached.");
  }
  const payload = await readJson(response);
  if (response.status === 401 || response.status === 403) {
    showLogin("Your session expired. Sign in again.");
    throw new AuthenticationError("Session expired.");
  }
  if (!response.ok || Array.isArray(payload.errors)) {
    const messages = Array.isArray(payload.errors)
      ? payload.errors.map((error) => error.message).filter(Boolean).join(" · ")
      : "";
    throw new Error(messages || errorMessage(payload, `Request failed with status ${response.status}.`));
  }
  if (!payload.data || typeof payload.data !== "object") throw new Error("The API returned an invalid response.");
  return payload.data;
}

function setupNavigationObserver() {
  const links = [...document.querySelectorAll(".nav-link")];
  if (!("IntersectionObserver" in window)) return;
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => {
      const active = link.getAttribute("href") === `#${visible.target.id}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-20% 0px -65%", threshold: [0, 0.1, 0.5] });
  links.forEach((link) => {
    const section = document.querySelector(link.getAttribute("href"));
    if (section) observer.observe(section);
  });
}

function upsertRepository(repository) {
  state.repositories = [repository, ...state.repositories.filter((item) => item.id !== repository.id)];
}

function upsertJob(job) {
  state.jobs = orderedRecentJobs([job, ...state.jobs.filter((item) => item.id !== job.id)]);
  state.jobHistoryPage = 1;
}

function repositoryName(repositoryId) {
  const repository = state.repositories.find((item) => item.id === repositoryId);
  return repository ? repository.name : repositoryId;
}

function setButtonPending(button, pending, pendingLabel = "Working...") {
  if (pending) {
    button.dataset.originalLabel = button.textContent;
    button.textContent = pendingLabel;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalLabel || button.textContent;
    button.disabled = false;
    delete button.dataset.originalLabel;
  }
}

function emptyState(title, detail, isError) {
  const empty = element("div", `empty-state${isError ? " error-state" : ""}`);
  if (isError) empty.setAttribute("role", "alert");
  empty.append(
    element("span", "", isError ? "!" : "○"),
    element("h3", "", title),
    element("p", "", detail),
  );
  return empty;
}

function toast(message, type = "success") {
  const notice = element("div", `toast ${type}`);
  notice.setAttribute("role", type === "error" ? "alert" : "status");
  const close = element("button", "toast-close", "Dismiss");
  close.type = "button";
  close.setAttribute("aria-label", "Dismiss notification");
  close.addEventListener("click", () => notice.remove());
  notice.append(element("span", "", message), close);
  elements["toast-region"].append(notice);
  window.setTimeout(() => notice.remove(), type === "error" ? 10000 : 6000);
}

function compactObject(source) {
  return Object.fromEntries(
    Object.entries(source)
      .filter(([, value]) => typeof value === "string" ? Boolean(value.trim()) : value != null)
      .map(([key, value]) => [key, typeof value === "string" ? value.trim() : value]),
  );
}

function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function cssEscape(value) {
  return window.CSS && typeof window.CSS.escape === "function"
    ? window.CSS.escape(value)
    : value.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function titleCase(value) {
  return String(value || "unknown").replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function initials(name) {
  return String(name).split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "RE";
}

function compactNumber(value) {
  const number = Number(value) || 0;
  if (number < 1000) return NUMBER_FORMAT.format(number);
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(number);
}

function formatRelative(value) {
  if (!value) return "unknown time";
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "unknown time";
  const seconds = Math.round((timestamp - Date.now()) / 1000);
  if (Math.abs(seconds) < 60) return RELATIVE_TIME.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return RELATIVE_TIME.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return RELATIVE_TIME.format(hours, "hour");
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 30) return RELATIVE_TIME.format(days, "day");
  return formatDate(value);
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "unknown date" : date.toLocaleString();
}

async function openKnowledge(kind, repository = null) {
  state.knowledgeSource = { kind, repository };
  state.knowledgeMode = "human";
  const dialog = elements["knowledge-dialog"];
  if (!dialog.open) dialog.showModal();
  await loadKnowledge();
}

async function switchKnowledgeMode(mode) {
  if (!state.knowledgeSource || !["human", "evidence"].includes(mode)) return;
  if (state.knowledgeMode === mode && state.knowledgeText) return;
  state.knowledgeMode = mode;
  await loadKnowledge();
}

async function loadKnowledge() {
  const source = state.knowledgeSource;
  if (!source) return;
  const { kind, repository } = source;
  const isCentral = kind === "central";
  const humanReadable = state.knowledgeMode === "human";
  const requestId = ++state.knowledgeRequest;
  elements["knowledge-kicker"].textContent = `${isCentral ? "KNOWLEDGE / CENTRAL" : "KNOWLEDGE / REPOSITORY"} · ${humanReadable ? "READER" : "EVIDENCE"}`;
  elements["knowledge-title"].textContent = isCentral ? "Central knowledge base" : `${repository.name} knowledge base`;
  elements["knowledge-meta"].textContent = "Loading secure document...";
  elements["knowledge-content"].replaceChildren(loadingDocument());
  elements["knowledge-copy"].disabled = true;
  elements["knowledge-reader"].classList.toggle("active", humanReadable);
  elements["knowledge-reader"].setAttribute("aria-pressed", String(humanReadable));
  elements["knowledge-evidence"].classList.toggle("active", !humanReadable);
  elements["knowledge-evidence"].setAttribute("aria-pressed", String(!humanReadable));
  state.knowledgeText = "";
  const base = humanReadable ? "/api/knowledge/human" : "/api/knowledge";
  const endpoint = isCentral ? `${base}/central` : `${base}/repositories/${encodeURIComponent(repository.id)}`;
  try {
    const response = await fetch(endpoint, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    const payload = await readJson(response);
    if (requestId !== state.knowledgeRequest) return;
    if (response.status === 401 || response.status === 403) {
      showLogin("Your session expired. Sign in again.");
      return;
    }
    if (!response.ok) {
      const fallback = response.status === 404
        ? humanReadable
          ? "The reader edition has not been published yet. Switch to Evidence or publish the reader editions."
          : "No knowledge document has been published for this source yet. Run an index and try again."
        : response.status === 409
          ? "The knowledge document is not ready while indexing is in progress."
          : "The knowledge document could not be loaded.";
      throw new Error(errorMessage(payload, fallback));
    }
    if (typeof payload.markdown !== "string") throw new Error("The knowledge response did not contain Markdown content.");
    state.knowledgeText = payload.markdown;
    elements["knowledge-title"].textContent = payload.title || elements["knowledge-title"].textContent;
    elements["knowledge-meta"].textContent = knowledgeMetadata(payload);
    renderMarkdown(elements["knowledge-content"], payload.markdown);
    elements["knowledge-copy"].disabled = false;
    elements["knowledge-content"].focus();
  } catch (error) {
    if (requestId !== state.knowledgeRequest) return;
    elements["knowledge-meta"].textContent = "Document unavailable";
    elements["knowledge-content"].replaceChildren(emptyState("Unable to open knowledge", messageFrom(error), true));
  }
}

function loadingDocument() {
  const loading = element("div", "document-loading");
  loading.setAttribute("role", "status");
  loading.append(element("span", "loading-mark"), element("p", "", "Decrypting and loading knowledge..."));
  return loading;
}

function knowledgeMetadata(payload) {
  const parts = [];
  if (payload.generatedAt) parts.push(`Generated ${formatDate(payload.generatedAt)}`);
  if (Number.isFinite(payload.sizeBytes)) parts.push(formatBytes(payload.sizeBytes));
  if (payload.scanRunId) parts.push(`scan ${shortId(payload.scanRunId)}`);
  if (payload.contentSha256) parts.push(`sha256 ${String(payload.contentSha256).slice(0, 12)}`);
  return parts.join(" · ") || "Generated knowledge document";
}

async function copyKnowledge() {
  if (!state.knowledgeText) return;
  try {
    await navigator.clipboard.writeText(state.knowledgeText);
    toast("Knowledge document copied to clipboard.");
  } catch (_error) {
    toast("Clipboard access was blocked by the browser.", "error");
  }
}

async function askArchitecture(event) {
  event.preventDefault();
  if (state.chatBusy) return;
  const question = elements["chat-question"].value.trim();
  if (!question) {
    elements["chat-error"].textContent = "Enter an architecture question.";
    elements["chat-question"].focus();
    return;
  }
  const history = state.chatHistory.slice(-8).map((message) => ({
    role: message.role,
    content: message.content,
  }));
  const input = compactObject({
    question,
    repositoryId: elements["chat-scope"].value,
    history,
  });
  state.chatBusy = true;
  elements["chat-error"].textContent = "";
  elements["chat-question"].disabled = true;
  setButtonPending(elements["chat-send"], true, "Thinking...");
  renderChatPending(question);
  try {
    const data = await graphqlRequest(ARCHITECTURE_CHAT_MUTATION, { input });
    const answer = data.askArchitecture;
    if (!answer || typeof answer.answerMarkdown !== "string") {
      throw new Error("The architecture agent returned an invalid response.");
    }
    state.chatHistory.push(
      { role: "user", content: question },
      {
        role: "assistant",
        content: answer.answerMarkdown,
        sources: Array.isArray(answer.sources) ? answer.sources : [],
        limitations: Array.isArray(answer.limitations) ? answer.limitations : [],
        invocation: answer.invocation || null,
      },
    );
    elements["chat-question"].value = "";
    renderArchitectureChat();
  } catch (error) {
    renderArchitectureChat();
    if (!(error instanceof AuthenticationError)) {
      elements["chat-error"].textContent = messageFrom(error);
    }
  } finally {
    state.chatBusy = false;
    elements["chat-question"].disabled = false;
    setButtonPending(elements["chat-send"], false);
    elements["chat-question"].focus();
  }
}

function clearArchitectureChat() {
  state.chatHistory = [];
  elements["chat-error"].textContent = "";
  renderArchitectureChat();
}

function renderArchitectureChat() {
  const container = elements["chat-messages"];
  container.replaceChildren();
  if (!state.chatHistory.length) {
    const welcome = element("div", "chat-welcome");
    const mark = element("span", "", "AI");
    mark.setAttribute("aria-hidden", "true");
    const body = element("div");
    body.append(
      element("h3", "", "Grounded answers from knowledge + Neo4j"),
      element("p", "", "Ask about service boundaries, functions, endpoints, events, database usage, or cross-service dependencies. Answers distinguish reader sections from exact graph evidence."),
    );
    welcome.append(mark, body);
    container.append(welcome);
    return;
  }
  state.chatHistory.forEach((message) => container.append(chatMessage(message)));
  container.scrollTop = container.scrollHeight;
}

function renderChatPending(question) {
  renderArchitectureChat();
  elements["chat-messages"].append(
    chatMessage({ role: "user", content: question }),
    chatMessage({ role: "assistant", content: "Searching the latest knowledge and Neo4j graph…", pending: true }),
  );
  elements["chat-messages"].scrollTop = elements["chat-messages"].scrollHeight;
}

function chatMessage(message) {
  const article = element("article", `chat-message ${message.role}${message.pending ? " pending" : ""}`);
  const label = element("span", "chat-role", message.role === "assistant" ? "GPT-OSS" : "YOU");
  const body = element("div", "chat-message-body");
  if (message.role === "assistant" && !message.pending) renderMarkdown(body, message.content);
  else body.append(element("p", "", message.content));
  article.append(label, body);
  if (message.role === "assistant" && Array.isArray(message.sources) && message.sources.length) {
    const sources = element("div", "chat-sources");
    sources.append(element("strong", "", "Sources used"));
    message.sources.forEach((source) => {
      const kind = source.evidenceKind === "graph" ? "graph" : "knowledge";
      sources.append(element(
        "span",
        `chat-source ${kind}`,
        `${source.id} · ${kind === "graph" ? "Neo4j graph" : "Knowledge"} · ${source.title} / ${source.section}`,
      ));
    });
    article.append(sources);
  }
  if (message.role === "assistant" && Array.isArray(message.limitations) && message.limitations.length) {
    const limitations = element("div", "chat-limitations");
    limitations.append(element("strong", "", "Limitations"));
    const list = element("ul");
    message.limitations.forEach((limitation) => list.append(element("li", "", limitation)));
    limitations.append(list);
    article.append(limitations);
  }
  if (message.role === "assistant" && message.invocation) {
    const total = Number(message.invocation.totalTokens) || 0;
    article.append(element(
      "small",
      "chat-telemetry",
      `${message.invocation.modelId || "Bedrock model"} · ${NUMBER_FORMAT.format(total)} tokens · ${(Number(message.invocation.latencyMs || 0) / 1000).toFixed(1)}s`,
    ));
  }
  return article;
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function shortId(value) {
  const text = String(value);
  return text.length > 16 ? `${text.slice(0, 8)}…${text.slice(-5)}` : text;
}

function renderMarkdown(container, markdown) {
  container.replaceChildren();
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    if (line.trimStart().startsWith("```")) {
      const language = line.trim().slice(3).trim();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trimStart().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const pre = element("pre");
      pre.append(element("code", language ? `language-${safeClassName(language)}` : "", codeLines.join("\n")));
      container.append(pre);
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line.trim());
    if (heading) {
      const node = element(`h${heading[1].length}`);
      appendInlineContent(node, heading[2]);
      container.append(node);
      index += 1;
      continue;
    }

    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      container.append(element("hr"));
      index += 1;
      continue;
    }

    const unordered = /^\s*[-*+]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      const orderedList = Boolean(ordered);
      const list = element(orderedList ? "ol" : "ul");
      while (index < lines.length) {
        const match = orderedList
          ? /^\s*\d+[.)]\s+(.+)$/.exec(lines[index])
          : /^\s*[-*+]\s+(.+)$/.exec(lines[index]);
        if (!match) break;
        const item = element("li");
        appendInlineContent(item, match[1]);
        list.append(item);
        index += 1;
      }
      container.append(list);
      continue;
    }

    if (/^>\s?/.test(line.trimStart())) {
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trimStart())) {
        quoteLines.push(lines[index].trimStart().replace(/^>\s?/, ""));
        index += 1;
      }
      const quote = element("blockquote");
      appendInlineContent(quote, quoteLines.join(" "));
      container.append(quote);
      continue;
    }

    if (index + 1 < lines.length && line.includes("|") && isTableDivider(lines[index + 1])) {
      const tableLines = [line];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        tableLines.push(lines[index]);
        index += 1;
      }
      container.append(markdownTable(tableLines));
      continue;
    }

    const paragraphLines = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines, index)) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = element("p");
    appendInlineContent(paragraph, paragraphLines.join(" "));
    container.append(paragraph);
  }
  if (!container.childNodes.length) container.append(emptyState("Empty document", "This knowledge document contains no readable content.", false));
}

function isMarkdownBlockStart(lines, index) {
  const line = lines[index];
  return /^#{1,4}\s+/.test(line.trim())
    || line.trimStart().startsWith("```")
    || /^\s*[-*+]\s+/.test(line)
    || /^\s*\d+[.)]\s+/.test(line)
    || /^>\s?/.test(line.trimStart())
    || /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)
    || (index + 1 < lines.length && line.includes("|") && isTableDivider(lines[index + 1]));
}

function appendInlineContent(parent, text) {
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("`")) {
      parent.append(element("code", "", token.slice(1, -1)));
    } else if (token.startsWith("**")) {
      parent.append(element("strong", "", token.slice(2, -2)));
    } else {
      const linkMatch = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      const safeUrl = linkMatch && safeHttpUrl(linkMatch[2]);
      if (linkMatch && safeUrl) {
        const link = element("a", "", linkMatch[1]);
        link.href = safeUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        parent.append(link);
      } else {
        parent.append(document.createTextNode(token));
      }
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function markdownTable(lines) {
  const wrapper = element("div", "markdown-table-scroll");
  const table = element("table");
  const head = element("thead");
  const body = element("tbody");
  const headerRow = element("tr");
  splitTableRow(lines[0]).forEach((cell) => {
    const heading = element("th");
    appendInlineContent(heading, cell);
    headerRow.append(heading);
  });
  head.append(headerRow);
  lines.slice(1).forEach((line) => {
    const row = element("tr");
    splitTableRow(line).forEach((cell) => {
      const value = element("td");
      appendInlineContent(value, cell);
      row.append(value);
    });
    body.append(row);
  });
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}

function splitTableRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function safeHttpUrl(value) {
  try {
    const url = new URL(value, window.location.origin);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch (_error) {
    return null;
  }
}

function safeClassName(value) {
  return value.toLowerCase().replace(/[^a-z0-9_-]/g, "").slice(0, 30);
}

async function readJson(response) {
  try { return await response.json(); } catch (_error) { return {}; }
}

function errorMessage(payload, fallback) {
  if (typeof payload.detail === "string") return payload.detail;
  if (payload.detail && typeof payload.detail.message === "string") return payload.detail.message;
  if (typeof payload.message === "string") return payload.message;
  if (Array.isArray(payload.errors) && payload.errors[0] && payload.errors[0].message) return payload.errors[0].message;
  return fallback;
}

function messageFrom(error) {
  return error instanceof Error ? error.message : "Something went wrong.";
}
