"""Generate Viggle v0.2.1 workflows using only built-in ComfyUI nodes."""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LORA = "qwen_image_2.1_viggle_turbo_v0.2.1_r256_comfy.safetensors"
RAW_SIGMAS = (0.9375, 0.875, 0.75, 0.5, 0.25)


def api(edit=False):
    def entry(kind, **inputs):
        # V3 Autogrow inputs are dotted keys in API prompts; ComfyUI rebuilds
        # the nested dictionary only when executing the node.
        flat = {}
        for key, value in inputs.items():
            if isinstance(value, dict):
                flat.update({f"{key}.{name}": item for name, item in value.items()})
            else:
                flat[key] = value
        return {"class_type": kind, "inputs": flat}
    graph = {
        "1": entry("UNETLoader", unet_name="qwen_image_2.1_bf16.safetensors", weight_dtype="default"),
        "2": entry("LoraLoaderBypassModelOnly", model=["1", 0], lora_name=LORA, strength_model=1.0),
        "3": entry("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default"),
        "4": entry("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "5": entry("TextEncodeQwenImage21", clip=["3", 0], vae=["4", 0],
                   prompt="Turn the red square into a yellow circle" if edit else "A red fox in a snowy forest",
                   negative_prompt="", resolution=512),
        "6": entry("StringFormat", values={"a": ["5", 2]},
                   f_string="a ** (0.5 + 0.4 * ({a[samples].shape[2]} * {a[samples].shape[3]} - 256) / 7936)"),
        "7": entry("ComfyMathExpression", expression=["6", 0], values={"a": ["24", 0]}),
        "13": entry("StringFormat", values={chr(97+i): [str(8+i), 0] for i in range(5)},
                    f_string="1, {a:.10f}, {b:.10f}, {c:.10f}, {d:.10f}, {e:.10f}, 0"),
        "14": entry("ManualSigmas", sigmas=["13", 0]),
        "15": entry("BasicGuider", model=["23", 0], conditioning=["5", 0]),
        "16": entry("RandomNoise", noise_seed=42),
        "17": entry("KSamplerSelect", sampler_name="euler"),
        "18": entry("SamplerCustomAdvanced", noise=["16", 0], guider=["15", 0], sampler=["17", 0],
                    sigmas=["14", 0], latent_image=["5", 2]),
        "19": entry("VAEDecode", samples=["18", 0], vae=["4", 0]),
        "20": entry("SaveImage", images=["19", 0], filename_prefix="ViggleV021"),
        "23": entry("QwenImage21Cache", model=["2", 0], device="off", dtype="default"),
        "24": entry("PrimitiveFloat", value=2.718281828459045),
    }
    for index, sigma in enumerate(RAW_SIGMAS, 8):
        graph[str(index)] = entry("ComfyMathExpression", expression=f"a / (a + 1 / {sigma} - 1)",
                                  values={"a": ["7", 0]})
    if edit:
        graph["21"] = entry("LoadImage", image="reference.png")
        graph["5"]["inputs"]["images.image_1"] = ["21", 0]
    return graph


# Input names/types, outputs, and widget order of the built-in nodes used above.
SPECS = {
    "UNETLoader": ([('unet_name','COMBO'), ('weight_dtype','COMBO')], [('MODEL','MODEL')], ['unet_name','weight_dtype']),
    "LoraLoaderBypassModelOnly": ([('model','MODEL'),('lora_name','COMBO'),('strength_model','FLOAT')], [('MODEL','MODEL')], ['lora_name','strength_model']),
    "CLIPLoader": ([('clip_name','COMBO'),('type','COMBO'),('device','COMBO')], [('CLIP','CLIP')], ['clip_name','type','device']),
    "VAELoader": ([('vae_name','COMBO')], [('VAE','VAE')], ['vae_name']),
    "TextEncodeQwenImage21": ([('clip','CLIP'),('vae','VAE'),('images.image_1','IMAGE'),('prompt','STRING'),('negative_prompt','STRING'),('resolution','INT')], [('positive','CONDITIONING'),('negative','CONDITIONING'),('latent','LATENT')], ['prompt','negative_prompt','resolution']),
    "StringFormat": ([], [('STRING','STRING')], ['f_string']),
    "ComfyMathExpression": ([('expression','STRING'),('values.a','FLOAT,INT,BOOLEAN')], [('FLOAT','FLOAT'),('INT','INT'),('BOOL','BOOLEAN')], ['expression']),
    "ManualSigmas": ([('sigmas','STRING')], [('SIGMAS','SIGMAS')], ['sigmas']),
    "BasicGuider": ([('model','MODEL'),('conditioning','CONDITIONING')], [('GUIDER','GUIDER')], []),
    "RandomNoise": ([('noise_seed','INT')], [('NOISE','NOISE')], ['noise_seed']),
    "KSamplerSelect": ([('sampler_name','COMBO')], [('SAMPLER','SAMPLER')], ['sampler_name']),
    "SamplerCustomAdvanced": ([('noise','NOISE'),('guider','GUIDER'),('sampler','SAMPLER'),('sigmas','SIGMAS'),('latent_image','LATENT')], [('output','LATENT'),('denoised_output','LATENT')], []),
    "VAEDecode": ([('samples','LATENT'),('vae','VAE')], [('IMAGE','IMAGE')], []),
    "SaveImage": ([('images','IMAGE'),('filename_prefix','STRING')], [], ['filename_prefix']),
    "LoadImage": ([('image','COMBO'),('upload','IMAGEUPLOAD')], [('IMAGE','IMAGE'),('MASK','MASK')], ['image']),
    "QwenImage21Cache": ([('model','MODEL'),('device','COMBO'),('dtype','COMBO')], [('MODEL','MODEL')], ['device','dtype']),
    "PrimitiveFloat": ([('value','FLOAT')], [('FLOAT','FLOAT')], ['value']),
}
POSITIONS = {1:(0,0), 2:(390,0), 3:(0,240), 4:(0,480), 5:(780,0),
             6:(0,1020), 7:(390,1020), 13:(1740,1290), 14:(1350,1290),
             15:(1240,0), 16:(1240,230), 17:(1240,420), 18:(1620,0),
             19:(2000,0), 20:(2380,0), 21:(390,320), 23:(780,-220), 24:(0,800)}
POSITIONS.update({i:(780 + (i-8)*350,1020) for i in range(8,13)})


def ui(edit=False):
    prompt = api(edit)
    nodes, links, by_id = [], [], {}
    flat_inputs = {}
    for identifier, item in sorted(prompt.items(), key=lambda pair: int(pair[0])):
        kind, args = item['class_type'], item['inputs']
        inputs, outputs, widget_names = SPECS[kind]
        flat = {}
        for key, value in args.items():
            if isinstance(value, dict):
                flat.update({f"{key}.{name}": data for name, data in value.items()})
            else:
                flat[key] = value
        flat_inputs[identifier] = flat
        if kind == "StringFormat":
            inputs = [(name, '*') for name in flat if name.startswith('values.')]+[('f_string','STRING')]
        node = {"id":int(identifier), "type":kind, "pos":list(POSITIONS[int(identifier)]),
                "size":[420 if kind == 'TextEncodeQwenImage21' else 320, 260 if kind in ('TextEncodeQwenImage21','LoadImage','SaveImage') else 150],
                "flags":{}, "order":len(nodes), "mode":0,
                "inputs":[{"name":name,"type":typ,"link":None,
                           **({"widget":{"name":name}} if name in widget_names else {})} for name,typ in inputs],
                "outputs":[{"name":name,"type":typ,"links":[]} for name,typ in outputs],
                "properties":{"Node name for S&R":kind,"cnr_id":"comfy-core","ver":"0.37.0"},
                "widgets_values":[flat.get(name) if not isinstance(flat.get(name),list) else '' for name in widget_names]}
        if kind == 'RandomNoise':
            node['widgets_values'].append('fixed')
        if kind == 'LoadImage':
            node['widgets_values'].append('image')
        nodes.append(node)
        by_id[identifier] = node
    for identifier, node in by_id.items():
        for slot, entry in enumerate(node['inputs']):
            value = flat_inputs[identifier].get(entry['name'])
            if not isinstance(value, list):
                continue
            source, source_slot = value
            output = by_id[source]['outputs'][source_slot]
            link_id = len(links)+1
            links.append([link_id,int(source),source_slot,int(identifier),slot,output['type']])
            entry['link'] = link_id
            output['links'].append(link_id)
    return {"id":str(uuid.uuid5(uuid.NAMESPACE_URL,f"t8-viggle-v021-native:{edit}")),
            "revision":0,"last_node_id":24,"last_link_id":len(links),"nodes":nodes,"links":links,
            "groups":[{"title":"Six-step schedule / 六步调度 · automatic resolution shift", "bounding":[-40,730,2620,760],"color":"#3f789e","font_size":24,"flags":{}}],
            "config":{},"extra":{},"version":0.4}


if __name__ == '__main__':
    for name, edit in (("text_to_image",False),("image_edit",True)):
        for folder, suffix, data in (("api_workflows","api",api(edit)),("workflows","ui",ui(edit))):
            path = ROOT / folder / f"viggle_v021_{name}_{suffix}.json"
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
