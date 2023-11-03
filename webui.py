import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.DataFrame.update(value=models_table), gr.CheckboxGroup.update(choices=tags)


def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'Não foi encontrado .pth no arquivo, confira no {extraction_folder}.')

    # move model and index file to extraction folder
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Baixando modelo de voz com nome: {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Pasta do modelo {dir_name} Já existe! Escolha um novo nome.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Extraindo zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Modelo baixado com sucesso!'

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Pasta do modelo {dir_name} already exists! Já existe! Escolha um novo nome.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] xtraindo zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Modelo enviado com sucesso!'

    except Exception as e:
        raise gr.Error(str(e))


def filter_models(tags, query):
    models_table = []

    # no filter
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on tags and query
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only tags
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # filter based on only query
    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    return gr.DataFrame.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])


def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='AICoverGenWebUI') as app:

        gr.Label('AICoverGenWebUI (adaptado para AI Hub Brasil por MrM0dZ)', show_label=False)

        # main tab
        with gr.Tab("Inferencia"):

            with gr.Accordion('Opções Principais'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Modelo de voz', info='Pasta de modelos: "AICoverGen --> rvc_models". Após novos modelos serem adicionados, clique no botão para atualizar.')
                        ref_btn = gr.Button('Atualizar Modelos 🔁', variant='primary')

                    with gr.Column() as yt_link_col:
                        song_input = gr.Text(label='Entrada de áudio', info='Link para uma do youtube ou caminho do arquivo local. Para enviar um arquivo, clique no botão abaixo.')
                        show_file_upload_button = gr.Button('Enviar arquivo')

                    with gr.Column(visible=False) as file_upload_col:
                        local_file = gr.File(label='Arquivo de áudio')
                        song_input_file = gr.UploadButton('Upload 📂', file_types=['audio'], variant='primary')
                        show_yt_link_button = gr.Button('Cole o link/caminho do arquivo.')
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-20, 20, value=0, step=1, label='Pitch (SOMENTE VOZ)', info='Use 12 para masculino > feminino e -12 para o inverso.')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Mudança geral de Pitch', info='Muda o pitch do áudio por completo, pode diminuir a qualidade.')
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Opções de conversão de voz', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Taxa do Index', info="Controla o sotaque do 'index' no áudio.")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Alcance de filtro', info="Se >=3: Aplica filtro para retirar 'respiração' de audios")
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS mix rate', info="Controla quanto imitar o volume original da voz (0) ou volume fixo (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect rate', info='Protege consoantes e sons de respiração. Utilize 0.5 para desativar.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Algoritmo de Pitch', info='Utilize rmvpe ou mangio-crepe para maior qualidade.')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe hop length', info='Valores menores alteram o pitch mais rapido. Pode quebrar a voz.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                keep_files = gr.Checkbox(label='Manter arquivos intermediarios.', info='Mantem todos arquivos gerados em song_output/id directory, e.g. Vozes/Instrumentais isoladas para agilizar processo.')

            with gr.Accordion('Opções de mixagem de áudio.', open=False):
                gr.Markdown('### Mudança de Volume (db)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Vocal Principal')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Back Vocal')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Musica')

                gr.Markdown('### Controle de Reverb na voz')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Tamanho da sala', info='Quanto maior, mais reverb.')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wetness', info='Nivel das vocais com reverb.')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dryness', info='Nivel das vocais sem reverb.')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Amortecimento', info='Absorção de frequencias altas no reverb.')

                gr.Markdown('### Formato do arquivo de saída.')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Tipo do arquivo:', info='mp3: pequeno, qualidade decente. wav: Grande, qualidade ótima.')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Resetar', components=[song_input, rvc_model, keep_files, local_file])
                generate_btn = gr.Button("Generate", variant='primary')
                ai_cover = gr.Audio(label='AI Cover', show_share_button=False)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                       protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                       output_format],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, output_format, ai_cover])

        # Download tab
        with gr.Tab('Baixar Modelo'):

            with gr.Tab('URL do HuggingFace/Pixeldrain'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Link do download', info='Deve ser o .zip contendo o .pth e .index')
                    model_name = gr.Text(label='Nome do Modelo', info='Dê um nome para seu modelo.')

                with gr.Row():
                    download_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Output Message', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Exemplos:')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                    ],
                    [model_zip_link, model_name],
                    [],
                    download_online_model,
                )

            with gr.Tab('Do Índice Público'):

                gr.Markdown('## Como usar')
                gr.Markdown('- Clique Inciar Tabela pública')
                gr.Markdown('- Filtre modelos com tags ou barra de pesquisa')
                gr.Markdown('- Selecione um modelo para preencher automaticamente.')
                gr.Markdown('- Clique Download')

                with gr.Row():
                    pub_zip_link = gr.Text(label='Link de download do modelo')
                    pub_model_name = gr.Text(label='Nome do modelo')

                with gr.Row():
                    download_pub_btn = gr.Button('Download 🌐', variant='primary', scale=19)
                    pub_dl_output_message = gr.Text(label='Mensagem de saída', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='Mostrar modelos com tags:', choices=[])
                search_query = gr.Text(label='Pesquisa')
                load_public_models_button = gr.Button(value='Inciar Tabela pública', variant='primary')

                public_models_table = gr.DataFrame(value=[], headers=['Model Name', 'Description', 'Credit', 'URL', 'Tags'], label='Modelos Públicos:', interactive=False)
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

        # Upload tab
        with gr.Tab('Enviar Modelo'):
            gr.Markdown('## Enviar modelo treinado com .pth e .index')
            gr.Markdown('- Encontre o arquivo do modelo (pasta weights) e arquivo opcional do index (pasta logs/[nome])')
            gr.Markdown('- Comprimir arquivos em .zip')
            gr.Markdown('- Enviar .zip e dar um nome para voz.')
            gr.Markdown('- Clique Enviar Modelo')

            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(label='Arquivo .zip')

                local_model_name = gr.Text(label='Nome Modelo')

            with gr.Row():
                model_upload_button = gr.Button('Enviar Modelo', variant='primary', scale=19)
                local_upload_output_message = gr.Text(label='Mensagem de saida', interactive=False, scale=20)
                model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(
        share=args.share_enabled,
        enable_queue=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
