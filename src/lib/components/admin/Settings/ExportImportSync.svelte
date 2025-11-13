<script lang="ts">
	import fileSaver from 'file-saver';
	const { saveAs } = fileSaver;

	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { exportAllSettings, importAllSettings, pushSyncData, pullSyncData } from '$lib/apis/configs';
	
	const i18n = getContext('i18n');

	export let saveHandler: Function;

	let importFileInput: HTMLInputElement;
	let syncInstanceUrl = '';
	let syncInstanceToken = '';
	let syncInProgress = false;
	let mergeOnImport = true;

	const exportAllHandler = async () => {
		try {
			const blob = await exportAllSettings(localStorage.token);
			if (blob) {
				const filename = `openwebui-settings-export-${Date.now()}.json`;
				saveAs(blob, filename);
				toast.success($i18n.t('Settings exported successfully'));
			}
		} catch (error) {
			toast.error($i18n.t('Failed to export settings: ') + error);
		}
	};

	const importAllHandler = (e: Event) => {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;

		const reader = new FileReader();
		reader.onload = async (e) => {
			try {
				const data = JSON.parse(e.target?.result as string);
				const res = await importAllSettings(localStorage.token, data, mergeOnImport);
				
				if (res && res.success) {
					toast.success($i18n.t('Settings imported successfully'));
					if (res.results.errors.length > 0) {
						console.warn('Import errors:', res.results.errors);
						toast.warning($i18n.t('Some items failed to import. Check console for details.'));
					}
				}
			} catch (error) {
				toast.error($i18n.t('Failed to import settings: ') + error);
			}
			(e.target as HTMLInputElement).value = '';
		};
		reader.readAsText(file);
	};

	const pushSyncHandler = async () => {
		if (!syncInstanceUrl || !syncInstanceToken) {
			toast.error($i18n.t('Please enter both instance URL and token'));
			return;
		}

		syncInProgress = true;
		try {
			const res = await pushSyncData(localStorage.token, syncInstanceUrl, syncInstanceToken);
			if (res && res.success) {
				toast.success($i18n.t('Successfully synced to remote instance'));
			}
		} catch (error) {
			toast.error($i18n.t('Failed to push sync: ') + error);
		} finally {
			syncInProgress = false;
		}
	};

	const pullSyncHandler = async () => {
		if (!syncInstanceUrl || !syncInstanceToken) {
			toast.error($i18n.t('Please enter both instance URL and token'));
			return;
		}

		syncInProgress = true;
		try {
			const res = await pullSyncData(localStorage.token, syncInstanceUrl, syncInstanceToken);
			if (res && res.success) {
				toast.success($i18n.t('Successfully synced from remote instance'));
			}
		} catch (error) {
			toast.error($i18n.t('Failed to pull sync: ') + error);
		} finally {
			syncInProgress = false;
		}
	};

	onMount(async () => {
		// Load saved sync settings if any
		const savedUrl = localStorage.getItem('syncInstanceUrl');
		const savedToken = localStorage.getItem('syncInstanceToken');
		if (savedUrl) syncInstanceUrl = savedUrl;
		if (savedToken) syncInstanceToken = savedToken;
	});

	$: {
		// Save sync settings
		if (syncInstanceUrl) localStorage.setItem('syncInstanceUrl', syncInstanceUrl);
		if (syncInstanceToken) localStorage.setItem('syncInstanceToken', syncInstanceToken);
	}
</script>

<form
	class="flex flex-col h-full justify-between space-y-3 text-sm"
	on:submit|preventDefault={() => saveHandler()}
>
	<div class="space-y-3 overflow-y-scroll scrollbar-hidden h-full">
		<!-- Export/Import Section -->
		<div>
			<div class="mb-2 text-sm font-medium">{$i18n.t('Export & Import')}</div>
			<div class="space-y-2">
				<!-- Export All Settings -->
				<button
					type="button"
					class="flex rounded-md py-2 px-3 w-full hover:bg-gray-200 dark:hover:bg-gray-800 transition"
					on:click={exportAllHandler}
				>
					<div class="self-center mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 16 16"
							fill="currentColor"
							class="w-4 h-4"
						>
							<path
								d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z"
							/>
							<path
								d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z"
							/>
						</svg>
					</div>
					<div class="self-center">{$i18n.t('Export All Settings')}</div>
				</button>

				<!-- Import All Settings -->
				<input
					bind:this={importFileInput}
					hidden
					type="file"
					accept=".json"
					on:change={importAllHandler}
				/>
				
				<div class="space-y-2">
					<button
						type="button"
						class="flex rounded-md py-2 px-3 w-full hover:bg-gray-200 dark:hover:bg-gray-800 transition"
						on:click={() => importFileInput?.click()}
					>
						<div class="self-center mr-3">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								viewBox="0 0 16 16"
								fill="currentColor"
								class="w-4 h-4"
							>
								<path
									d="M8.75 13.25a.75.75 0 0 1-1.5 0V7.56L5.03 9.78a.75.75 0 0 1-1.06-1.06l3.5-3.5a.75.75 0 0 1 1.06 0l3.5 3.5a.75.75 0 1 1-1.06 1.06L8.75 7.56v5.69Z"
								/>
								<path
									d="M3.5 6.25a.75.75 0 0 1-1.5 0v-1.5A2.75 2.75 0 0 1 4.75 2h6.5A2.75 2.75 0 0 1 14 4.75v1.5a.75.75 0 0 1-1.5 0v-1.5c0-.69-.56-1.25-1.25-1.25h-6.5c-.69 0-1.25.56-1.25 1.25v1.5Z"
								/>
							</svg>
						</div>
						<div class="self-center">{$i18n.t('Import All Settings')}</div>
					</button>
					
					<label class="flex items-center space-x-2 pl-3">
						<input type="checkbox" bind:checked={mergeOnImport} class="rounded" />
						<span class="text-xs">{$i18n.t('Merge with existing settings (uncheck to replace)')}</span>
					</label>
				</div>

				<div class="text-xs text-gray-500 dark:text-gray-400 pl-3">
					{$i18n.t('Export includes all custom models, tools, functions, and prompts. Import creates automatic backup.')}
				</div>
			</div>
		</div>

		<!-- LAN Sync Section -->
		<div>
			<div class="mb-2 text-sm font-medium">{$i18n.t('LAN Synchronization')}</div>
			<div class="space-y-3">
				<div>
					<label class="text-xs text-gray-500 dark:text-gray-400">
						{$i18n.t('Remote Instance URL')}
					</label>
					<input
						type="text"
						bind:value={syncInstanceUrl}
						placeholder="http://192.168.1.50:3000"
						class="w-full rounded-md py-2 px-3 text-sm bg-transparent border border-gray-300 dark:border-gray-600"
					/>
				</div>

				<div>
					<label class="text-xs text-gray-500 dark:text-gray-400">
						{$i18n.t('Remote Instance Token')}
					</label>
					<input
						type="password"
						bind:value={syncInstanceToken}
						placeholder="sk-..."
						class="w-full rounded-md py-2 px-3 text-sm bg-transparent border border-gray-300 dark:border-gray-600"
					/>
				</div>

				<div class="flex space-x-2">
					<button
						type="button"
						class="flex-1 rounded-md py-2 px-3 bg-blue-500 hover:bg-blue-600 text-white transition disabled:opacity-50"
						on:click={pushSyncHandler}
						disabled={syncInProgress}
					>
						{syncInProgress ? $i18n.t('Syncing...') : $i18n.t('Push to Remote')}
					</button>
					<button
						type="button"
						class="flex-1 rounded-md py-2 px-3 bg-green-500 hover:bg-green-600 text-white transition disabled:opacity-50"
						on:click={pullSyncHandler}
						disabled={syncInProgress}
					>
						{syncInProgress ? $i18n.t('Syncing...') : $i18n.t('Pull from Remote')}
					</button>
				</div>

				<div class="text-xs text-gray-500 dark:text-gray-400">
					{$i18n.t('Sync custom models, tools, functions, and prompts with other Open WebUI instances on your LAN. Only LAN addresses are allowed (192.168.x.x, 10.x.x.x, 172.x.x.x).')}
				</div>
			</div>
		</div>
	</div>
</form>
