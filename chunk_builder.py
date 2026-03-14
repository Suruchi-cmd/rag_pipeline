"""
chunk_builder.py — Voice-optimized chunking for all Google Sheets.

Each builder reads raw sheet data (list of dicts from gspread.get_all_records())
and returns a list of ChunkRecord objects with speakable answer text.

Every chunk must be a complete, speakable sentence or paragraph — suitable
for Felicia to read aloud on a phone call via TTS.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from models import ChunkRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_chunks_from_sheet(sheet_name: str, rows: list[dict]) -> list[ChunkRecord]:
    """Route to the appropriate builder based on sheet name."""
    builders = {
        'Location Info': build_location_chunks,
        'Jump Prices': build_jump_price_chunks,
        'Go Karting': build_go_karting_chunks,
        'Special Programs': build_special_program_chunks,
        'Attractions': build_attraction_chunks,
        'Birthday Parties': build_birthday_chunks,
        'Group Bookings': build_group_booking_chunks,
        'Aero Camp': build_camp_chunks,
        'Passes': build_passes_chunks,
        'Promotions': build_promotion_chunks,
        'Park Rules': build_park_rules_chunks,
        'FAQs': build_faq_chunks,
        'Voice Call Scripts': build_voice_script_chunks,
        'Chatbot Quick Replies': build_chatbot_qr_chunks,
    }
    builder = builders.get(sheet_name)
    if not builder:
        logger.warning("No builder for sheet %r — skipping", sheet_name)
        return []
    return builder(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_row(row: dict) -> bool:
    """Return True if the row is empty or a sub-table header."""
    if not row:
        return True
    # All values blank
    if all(str(v).strip() == '' for v in row.values()):
        return True
    # Sub-table header row (chunk_id column literally contains "chunk_id")
    chunk_id_val = str(row.get('chunk_id', '')).strip().lower()
    if chunk_id_val == 'chunk_id':
        return True
    return False


def _get(row: dict, key: str, default: str = '') -> str:
    """Get a stripped string value from a row dict."""
    return str(row.get(key, default)).strip()


def _price_spoken(price_str: str) -> str:
    """Convert a price string like '$24.90' to speakable form 'twenty four ninety'."""
    # Keep as-is for the answer text — the LLM/Felicia prompt already handles
    # phonetic conversion. Just ensure 'plus tax' is appended where needed.
    price_str = price_str.replace('$', '').strip()
    if price_str:
        return f"${price_str} plus tax"
    return ''


def _make_tags(sheet_name: str, *extra: str) -> list[str]:
    """Build a tags list from sheet name + extras."""
    base = sheet_name.lower().replace(' ', '_')
    tags = [base]
    for t in extra:
        t = t.strip().lower().replace(' ', '_')
        if t and t not in tags:
            tags.append(t)
    return tags


def _chunk(
    chunk_id: str,
    category: str,
    subcategory: str,
    question: str,
    answer: str,
    tags: list[str],
    sheet_name: str,
    source: str = 'knowledge_base',
    metadata: dict | None = None,
) -> ChunkRecord:
    """Convenience factory for ChunkRecord."""
    return ChunkRecord(
        id=chunk_id,
        category=category,
        subcategory=subcategory,
        location='Scarborough',
        question=question,
        answer=answer,
        tags=tags,
        sheet_name=sheet_name,
        source=source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Builders — one per sheet
# ---------------------------------------------------------------------------


def build_location_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Location Info → merge rows into 1-2 speakable chunks."""
    chunks: list[ChunkRecord] = []
    contact_parts: list[str] = []
    hours_parts: list[str] = []
    links_parts: list[str] = []

    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')

        if 'contact' in cid:
            # Gather contact details
            for key in ('address', 'city', 'province', 'postal_code', 'phone', 'email'):
                val = _get(row, key)
                if val:
                    contact_parts.append((key, val))
        elif 'hours' in cid:
            for key in ('days', 'hours', 'day', 'time'):
                val = _get(row, key)
                if val:
                    hours_parts.append((key, val))
        elif 'links' in cid:
            for key in row:
                val = _get(row, key)
                if val and key != 'chunk_id':
                    links_parts.append((key, val))

    # Build contact chunk from all contact rows
    if contact_parts or any(rows):
        # Collect all values from contact rows into a single spoken paragraph
        address_vals = {}
        for row in rows:
            if _skip_row(row):
                continue
            cid = _get(row, 'chunk_id')
            if 'contact' in cid:
                for k, v in row.items():
                    if str(v).strip() and k != 'chunk_id':
                        address_vals[k] = str(v).strip()

        parts = []
        addr = address_vals.get('address', '')
        city = address_vals.get('city', '')
        province = address_vals.get('province', '')
        postal = address_vals.get('postal_code', '')
        if addr:
            loc = f"AeroSports Scarborough is located at {addr}"
            if city:
                loc += f" in {city}"
            if province:
                loc += f", {province}"
            if postal:
                loc += f", postal code {postal}"
            loc += "."
            parts.append(loc)

        phone = address_vals.get('phone', '')
        email = address_vals.get('email', '')
        if phone and email:
            parts.append(f"You can reach us by phone at {phone} or by email at {email}.")
        elif phone:
            parts.append(f"You can reach us by phone at {phone}.")
        elif email:
            parts.append(f"You can reach us by email at {email}.")

        if parts:
            chunks.append(_chunk(
                chunk_id='scb_contact_001',
                category='Contact',
                subcategory='Contact Info',
                question='How do I contact AeroSports Scarborough?',
                answer=' '.join(parts),
                tags=_make_tags('Location Info', 'contact', 'address', 'phone', 'email'),
                sheet_name='Location Info',
            ))

    # Build hours chunk
    hours_vals: list[dict] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if 'hours' in cid:
            hours_vals.append(row)

    if hours_vals:
        time_parts = []
        for row in hours_vals:
            days = _get(row, 'days') or _get(row, 'day')
            time = _get(row, 'hours') or _get(row, 'time')
            if days and time:
                time_parts.append(f"{days} from {time}")
        if time_parts:
            answer = "AeroSports Scarborough is open " + ", and ".join(time_parts) + "."
            chunks.append(_chunk(
                chunk_id='scb_hours_002',
                category='Contact',
                subcategory='Hours',
                question='What are the hours for AeroSports Scarborough?',
                answer=answer,
                tags=_make_tags('Location Info', 'hours', 'schedule'),
                sheet_name='Location Info',
            ))

    # Build links chunk
    links_vals: list[dict] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if 'links' in cid:
            links_vals.append(row)

    if links_vals:
        link_parts = []
        for row in links_vals:
            for k, v in row.items():
                val = str(v).strip()
                if val and k != 'chunk_id':
                    link_parts.append(f"The {k.replace('_', ' ')} link is {val}.")
        if link_parts:
            chunks.append(_chunk(
                chunk_id='scb_links_003',
                category='Contact',
                subcategory='Links',
                question='What are the booking and waiver links?',
                answer=' '.join(link_parts),
                tags=_make_tags('Location Info', 'links', 'booking', 'waiver'),
                sheet_name='Location Info',
            ))

    return chunks


def build_jump_price_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Jump Prices → one chunk per pass type as a sentence."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        pass_type = _get(row, 'pass_type') or _get(row, 'type') or _get(row, 'name')
        price = _get(row, 'price')
        duration = _get(row, 'duration') or _get(row, 'time')
        ages = _get(row, 'ages') or _get(row, 'age') or _get(row, 'age_range')
        includes = _get(row, 'includes') or _get(row, 'description') or _get(row, 'details')
        notes = _get(row, 'notes') or _get(row, 'note')

        parts = []
        if pass_type and price:
            parts.append(f"The {pass_type} at AeroSports Scarborough is {_price_spoken(price)}")
            if duration:
                parts[-1] += f" for {duration} of jump time"
            parts[-1] += "."
        elif pass_type:
            parts.append(f"The {pass_type} at AeroSports Scarborough.")

        if ages:
            parts.append(f"It's available for {ages}.")
        if includes:
            parts.append(f"This pass includes {includes}.")
        if notes:
            parts.append(notes if notes.endswith('.') else notes + '.')

        # Socks info
        if 'socks' in cid.lower():
            sock_price = _get(row, 'price') or _get(row, 'cost')
            name = _get(row, 'name') or _get(row, 'item') or 'AeroSports grip socks'
            if sock_price:
                parts = [f"{name} are required and cost {_price_spoken(sock_price)}. They are reusable, so bring them back for future visits."]
            else:
                parts = [f"{name} are required for jumping. They are reusable, so bring them back for future visits."]

        if parts:
            answer = ' '.join(parts)
            chunks.append(_chunk(
                chunk_id=cid,
                category='Pricing',
                subcategory='Jump Passes' if 'socks' not in cid.lower() else 'Socks',
                question=f'How much does the {pass_type or "jump pass"} cost?',
                answer=answer,
                tags=_make_tags('Jump Prices', 'pricing', 'jump', pass_type.lower().replace(' ', '_') if pass_type else ''),
                sheet_name='Jump Prices',
            ))
    return chunks


def build_go_karting_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Go Karting → one chunk per track+race_type combo."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        track = _get(row, 'track')
        race_type = _get(row, 'race_type')
        price = _get(row, 'price')
        min_height = _get(row, 'min_height')
        notes = _get(row, 'notes')

        parts = []
        desc = f"A {race_type} go kart race" if race_type else "A go kart race"
        if track:
            desc += f" on the {track} track"
        if price:
            desc += f" costs {_price_spoken(price)}"
        desc += "."
        parts.append(desc)

        if race_type and 'standalone' in race_type.lower():
            parts.append("This is for guests who are not purchasing a jump pass.")
        elif race_type and 'add' in race_type.lower():
            parts.append("This is for guests who already have a jump pass.")

        if notes:
            parts.append(notes if notes.endswith('.') else notes + '.')
        if min_height:
            parts.append(f"Drivers must be at least {min_height} tall and closed-toed shoes are required.")

        answer = ' '.join(parts)
        chunks.append(_chunk(
            chunk_id=cid,
            category='Go Karting',
            subcategory=track or 'Go Karting',
            question=f'What are the go karting options and prices?',
            answer=answer,
            tags=_make_tags('Go Karting', 'go_kart', track.lower().replace(' ', '_') if track else ''),
            sheet_name='Go Karting',
            metadata={'track': track, 'race_type': race_type},
        ))
    return chunks


def build_special_program_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Special Programs → one chunk per program."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        program = _get(row, 'program') or _get(row, 'name')
        availability = _get(row, 'availability')
        time = _get(row, 'time')
        price = _get(row, 'price')
        details = _get(row, 'details') or _get(row, 'description')

        parts = []
        if program:
            intro = f"{program} at AeroSports Scarborough"
            if availability:
                intro += f" is available {availability}"
            if time:
                intro += f" from {time}"
            intro += "."
            parts.append(intro)
        if price:
            parts.append(f"It costs {_price_spoken(price)}.")
        if details:
            parts.append(details if details.endswith('.') else details + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Special Programs',
                subcategory=program or 'Special Programs',
                question=f'What is {program}?' if program else 'What special programs are available?',
                answer=' '.join(parts),
                tags=_make_tags('Special Programs', program.lower().replace(' ', '_') if program else ''),
                sheet_name='Special Programs',
            ))
    return chunks


def build_attraction_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Attractions → one chunk per attraction."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        name = _get(row, 'attraction_name') or _get(row, 'name')
        desc = _get(row, 'description')
        reqs = _get(row, 'age_height_requirements') or _get(row, 'requirements')
        included = _get(row, 'included_in_jump_pass')
        pricing_note = _get(row, 'pricing_note') or _get(row, 'notes')

        parts = []
        if name and desc:
            parts.append(f"{name} at AeroSports Scarborough {desc}")
            if not desc.endswith('.'):
                parts[-1] += '.'
        elif name:
            parts.append(f"{name} is an attraction at AeroSports Scarborough.")

        if reqs:
            parts.append(f"It is suitable for {reqs}." if 'all' in reqs.lower() else f"Requirements: {reqs}.")
        if included and included.lower() in ('yes', 'true'):
            parts.append(f"{name} is included with any jump pass at no extra cost.")
        elif included and included.lower() not in ('no', 'false', ''):
            parts.append(included if included.endswith('.') else included + '.')
        if pricing_note:
            parts.append(pricing_note if pricing_note.endswith('.') else pricing_note + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Attractions',
                subcategory=name or 'Attractions',
                question=f'What is {name}?' if name else 'What attractions are available?',
                answer=' '.join(parts),
                tags=_make_tags('Attractions', name.lower().replace(' ', '_') if name else ''),
                sheet_name='Attractions',
            ))
    return chunks


def build_birthday_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Birthday Parties → one chunk per package + add-ons + terms.

    Handles composite sheets with sub-tables (packages, add-ons, terms).
    """
    chunks: list[ChunkRecord] = []
    # Group rows by chunk_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue
        groups[cid].append(row)

    for cid, group_rows in groups.items():
        parts = []
        for row in group_rows:
            # Try to build a natural sentence from available fields
            name = _get(row, 'package') or _get(row, 'name') or _get(row, 'item') or _get(row, 'addon') or _get(row, 'add_on')
            price = _get(row, 'price') or _get(row, 'cost')
            jump_time = _get(row, 'jump_time') or _get(row, 'duration')
            room_time = _get(row, 'room_time')
            includes = _get(row, 'includes') or _get(row, 'description') or _get(row, 'details')
            min_jumpers = _get(row, 'min_jumpers') or _get(row, 'minimum')
            notes = _get(row, 'notes') or _get(row, 'terms')

            if name and price:
                sentence = f"The {name} is {_price_spoken(price)} per jumper"
                if jump_time:
                    sentence += f" with {jump_time} of jump time"
                if room_time:
                    sentence += f" and {room_time} in a private party room"
                sentence += "."
                parts.append(sentence)
            elif name:
                parts.append(f"The {name}.")

            if includes:
                parts.append(includes if includes.endswith('.') else includes + '.')
            if min_jumpers:
                parts.append(f"Minimum {min_jumpers} jumpers.")
            if notes:
                parts.append(notes if notes.endswith('.') else notes + '.')

        if parts:
            subcategory = 'Add-Ons' if 'addon' in cid.lower() else 'Birthday Packages'
            chunks.append(_chunk(
                chunk_id=cid,
                category='Birthday Parties',
                subcategory=subcategory,
                question='What are the birthday party packages?' if 'addon' not in cid.lower() else 'What birthday party add-ons are available?',
                answer=' '.join(parts),
                tags=_make_tags('Birthday Parties', 'birthday', 'party'),
                sheet_name='Birthday Parties',
            ))
    return chunks


def build_group_booking_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Group Bookings → one chunk per section (group, corporate, school, fundraise, facility, rooms)."""
    chunks: list[ChunkRecord] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue
        groups[cid].append(row)

    # Map chunk_id prefix to question/subcategory
    _questions = {
        'group': ('How do group bookings work?', 'Group Bookings'),
        'corporate': ('How do corporate events work?', 'Corporate Events'),
        'school': ('How do school field trips work?', 'School Trips'),
        'fundraise': ('How do fundraising events work?', 'Fundraising'),
        'facility': ('What does a facility rental include?', 'Facility Rentals'),
        'rooms': ('What are the party room options?', 'Room Rentals'),
    }

    for cid, group_rows in groups.items():
        parts = []
        for row in group_rows:
            name = _get(row, 'name') or _get(row, 'package') or _get(row, 'type') or _get(row, 'room')
            price = _get(row, 'price') or _get(row, 'cost') or _get(row, 'rate')
            capacity = _get(row, 'capacity') or _get(row, 'min_guests') or _get(row, 'group_size')
            includes = _get(row, 'includes') or _get(row, 'description') or _get(row, 'details')
            notes = _get(row, 'notes')
            duration = _get(row, 'duration') or _get(row, 'time')

            if name and price:
                sentence = f"{name} costs {_price_spoken(price)}"
                if capacity:
                    sentence += f" for {capacity} guests"
                if duration:
                    sentence += f" for {duration}"
                sentence += "."
                parts.append(sentence)
            elif name:
                parts.append(f"{name}.")

            if includes:
                parts.append(includes if includes.endswith('.') else includes + '.')
            if notes:
                parts.append(notes if notes.endswith('.') else notes + '.')

        if parts:
            # Determine subcategory from prefix
            prefix = cid.split('_')[1] if len(cid.split('_')) >= 2 else 'group'
            question, subcategory = _questions.get(prefix, ('How do group bookings work?', 'Group Bookings'))

            chunks.append(_chunk(
                chunk_id=cid,
                category='Group Bookings',
                subcategory=subcategory,
                question=question,
                answer=' '.join(parts),
                tags=_make_tags('Group Bookings', prefix),
                sheet_name='Group Bookings',
            ))
    return chunks


def build_camp_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Aero Camp → one chunk for pricing + details + add-ons."""
    chunks: list[ChunkRecord] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue
        groups[cid].append(row)

    for cid, group_rows in groups.items():
        parts = []
        for row in group_rows:
            name = _get(row, 'name') or _get(row, 'program') or _get(row, 'item')
            price = _get(row, 'price') or _get(row, 'cost')
            ages = _get(row, 'ages') or _get(row, 'age_range')
            time = _get(row, 'time') or _get(row, 'schedule')
            includes = _get(row, 'includes') or _get(row, 'description') or _get(row, 'details')
            notes = _get(row, 'notes')

            if name:
                sentence = f"{name}"
                if price:
                    sentence += f" costs {_price_spoken(price)}"
                if ages:
                    sentence += f" for ages {ages}"
                if time:
                    sentence += f", running {time}"
                sentence += "."
                parts.append(sentence)

            if includes:
                parts.append(includes if includes.endswith('.') else includes + '.')
            if notes:
                parts.append(notes if notes.endswith('.') else notes + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Aero Camp',
                subcategory='Aero Camp',
                question='What is Aero Camp?',
                answer=' '.join(parts),
                tags=_make_tags('Aero Camp', 'camp', 'kids'),
                sheet_name='Aero Camp',
            ))
    return chunks


def build_passes_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Passes → one chunk with all passes as paragraph."""
    chunks: list[ChunkRecord] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue
        groups[cid].append(row)

    for cid, group_rows in groups.items():
        parts = []
        for row in group_rows:
            name = _get(row, 'name') or _get(row, 'pass_name') or _get(row, 'type')
            price = _get(row, 'price') or _get(row, 'cost')
            details = _get(row, 'details') or _get(row, 'description') or _get(row, 'includes')
            validity = _get(row, 'validity') or _get(row, 'duration')
            notes = _get(row, 'notes')

            if name and price:
                sentence = f"The {name} is {_price_spoken(price)}"
                if validity:
                    sentence += f" and is valid for {validity}"
                sentence += "."
                parts.append(sentence)
            elif name:
                parts.append(f"The {name}.")

            if details:
                parts.append(details if details.endswith('.') else details + '.')
            if notes:
                parts.append(notes if notes.endswith('.') else notes + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Passes',
                subcategory='Passes',
                question='What passes are available?',
                answer=' '.join(parts),
                tags=_make_tags('Passes', 'pass', 'membership'),
                sheet_name='Passes',
            ))
    return chunks


def build_promotion_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Promotions → one chunk per active promo (skip status != Active)."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        status = _get(row, 'status')
        if status.lower() != 'active':
            continue

        promo_name = _get(row, 'promo_name') or _get(row, 'name')
        description = _get(row, 'description')
        code = _get(row, 'code')
        applies_to = _get(row, 'applies_to')
        discount = _get(row, 'discount')
        terms = _get(row, 'terms')

        parts = []
        if promo_name:
            parts.append(f"Right now we have a {promo_name}.")
        if code:
            parts.append(f"Use code {code}")
            if discount:
                parts[-1] += f" for {discount}"
            if applies_to:
                parts[-1] += f" on {applies_to}"
            parts[-1] += "."
        elif discount:
            parts.append(f"You get {discount}.")

        if description and description not in promo_name:
            parts.append(description if description.endswith('.') else description + '.')
        if terms:
            parts.append(terms if terms.endswith('.') else terms + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Promotions',
                subcategory='Promotions',
                question='Are there any current promotions or deals?',
                answer=' '.join(parts),
                tags=_make_tags('Promotions', 'promo', 'deal', code.lower() if code else ''),
                sheet_name='Promotions',
                metadata={'code': code, 'status': status},
            ))
    return chunks


def build_park_rules_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Park Rules → group related rules into 2-3 sentence chunks."""
    chunks: list[ChunkRecord] = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue
        groups[cid].append(row)

    for cid, group_rows in groups.items():
        parts = []
        topic = ''
        for row in group_rows:
            topic = _get(row, 'rule_topic') or topic
            rule_text = _get(row, 'rule_text')
            if rule_text:
                parts.append(rule_text if rule_text.endswith('.') else rule_text + '.')

        if parts:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Park Rules',
                subcategory=topic or 'General Rules',
                question=f'What are the park rules for {topic.lower()}?' if topic else 'What are the park rules?',
                answer=' '.join(parts),
                tags=_make_tags('Park Rules', 'rules', 'safety', topic.lower().replace(' ', '_') if topic else ''),
                sheet_name='Park Rules',
            ))
    return chunks


def build_faq_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """FAQs → one chunk per FAQ."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        category = _get(row, 'category') or 'FAQ'
        question = _get(row, 'question')
        answer = _get(row, 'answer')

        if question and answer:
            chunks.append(_chunk(
                chunk_id=cid,
                category='FAQ',
                subcategory=category,
                question=question,
                answer=answer,
                tags=_make_tags('FAQs', 'faq', category.lower().replace(' ', '_')),
                sheet_name='FAQs',
            ))
    return chunks


def build_voice_script_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Voice Call Scripts → one chunk per scenario (source = voice_script)."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        scenario = _get(row, 'scenario')
        script = _get(row, 'script')
        notes = _get(row, 'notes')

        if script:
            answer = script
            if notes:
                answer += ' ' + (notes if notes.endswith('.') else notes + '.')

            chunks.append(_chunk(
                chunk_id=cid,
                category='Voice Scripts',
                subcategory=scenario or 'General',
                question=f'Information for phone callers about {scenario.lower()}' if scenario else 'Information for phone callers',
                answer=answer,
                tags=_make_tags('Voice Call Scripts', 'voice', 'phone', scenario.lower().replace(' ', '_') if scenario else ''),
                sheet_name='Voice Call Scripts',
                source='voice_script',
                metadata={'scenario': scenario},
            ))
    return chunks


def build_chatbot_qr_chunks(rows: list[dict]) -> list[ChunkRecord]:
    """Chatbot Quick Replies → one chunk per intent (source = chatbot_qr)."""
    chunks: list[ChunkRecord] = []
    for row in rows:
        if _skip_row(row):
            continue
        cid = _get(row, 'chunk_id')
        if not cid:
            continue

        intent = _get(row, 'intent')
        sample_question = _get(row, 'sample_question')
        response = _get(row, 'response')

        if sample_question and response:
            chunks.append(_chunk(
                chunk_id=cid,
                category='Quick Replies',
                subcategory=intent or 'General',
                question=sample_question,
                answer=response,
                tags=_make_tags('Chatbot Quick Replies', 'chatbot', 'quick_reply', intent.lower().replace(' ', '_') if intent else ''),
                sheet_name='Chatbot Quick Replies',
                source='chatbot_qr',
                metadata={'intent': intent},
            ))
    return chunks
