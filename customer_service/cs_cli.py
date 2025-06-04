import click
import os
import requests
import sys
import shlex
import re

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://127.0.0.1:8003')


def do_upload(file_path: str):
    """Upload file to RAG service and echo kb_id."""
    url = f"{API_BASE_URL}/upload_faq"
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            resp = requests.post(url, files=files)
    except requests.RequestException as e:
        click.echo(f"Error connecting to service: {e}")
        return
    except FileNotFoundError:
        click.echo(f"File not found: {file_path}")
        return
    try:
        data = resp.json()
    except ValueError:
        click.echo(f"Invalid JSON response: {resp.text}")
        return
    if resp.status_code == 200 and 'kb_id' in data:
        click.echo(f"New KB created with id: {data['kb_id']}")
    else:
        msg = data.get('message', resp.text)
        click.echo(f"Upload failed: {msg}")


def do_chat(message: str, kb_id: str = None):
    """Send chat with optional KB ID and stream response."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        'query': message,
        'session_id': 'e',
        'use_kb': True
    }
    if kb_id:
        payload['kb_id'] = kb_id
    try:
        resp = requests.post(url, json=payload, stream=True)
    except requests.RequestException as e:
        click.echo(f"Error connecting to service: {e}")
        return
    if resp.status_code != 200:
        click.echo(f"Request failed: {resp.status_code} {resp.text}")
        return
    try:
        for chunk in resp.iter_content(chunk_size=512, decode_unicode=True):
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
    except Exception as e:
        click.echo(f"Streaming error: {e}")
        return
    # Ensure prompt on new line after response
    sys.stdout.write("\n")
    sys.stdout.flush()


@click.group(invoke_without_command=True)
@click.option('--kb-id', default=None, help='Optional KB ID to use for chat')
@click.argument('message', required=False)
@click.pass_context

def cli(ctx, kb_id, message):
    """CLI for RAG chat with simple interface."""
    ctx.ensure_object(dict)
    ctx.obj['kb_id'] = kb_id
    # No subcommand invoked, handle default behavior
    if ctx.invoked_subcommand is None:
        if message is None:
            click.echo(ctx.get_help())
        else:
            # If user passed 'shell', invoke interactive shell
            if message.strip() == 'shell':
                ctx.invoke(shell)
            else:
                do_chat(message, kb_id)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def upload(file_path):
    """Upload CSV/XLSX file and get KB ID."""
    do_upload(file_path)


@cli.command(name='shell')
def shell():
    """Interactive shell supporting chat with optional KB ID and upload commands"""
    click.echo("Entering interactive shell. Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            line = input('cs> ')
        except (EOFError, KeyboardInterrupt):
            click.echo('\nExiting shell.')
            break
        if not line.strip():
            continue
        parts = shlex.split(line)
        cmd = parts[0]
        if cmd in ('exit', 'quit'):
            click.echo('Goodbye!')
            break
        if cmd == 'help':
            click.echo("Commands: upload <file_path> | [kb_id=<kb_id>] <message>")
            continue
        if cmd == 'upload':
            if len(parts) != 2:
                click.echo('Usage: upload <file_path>')
                continue
            do_upload(parts[1])
            continue
        # Chat: support optional prefix for kb_id in shell
        match = re.match(r'kb_id=(\S+)\s+(.+)', line)
        if match:
            kb_val, msg = match.groups()
            do_chat(msg, kb_val)
        else:
            do_chat(line, None)


if __name__ == '__main__':
    cli()
