```markdown
# AeroSports Scarborough Understanding Document

## Overview of the Business and Dataset

AeroSports Scarborough is an entertainment facility offering a variety of activities such as trampolines, go-karting, laser tag, and birthday party packages. The dataset comprises several workbooks that provide detailed information about pricing, passes, promotions, voice call scripts, FAQs, and more. These documents are designed to support customer service operations by providing consistent responses and facilitating efficient booking processes.

## Passes

### Purpose
The "Passes" workbook outlines the different types of passes available at AeroSports Scarborough, including their durations and prices. It serves as a reference for both customers and staff to understand the options for accessing various activities within the facility.

### Key Entities, Columns, and Terminology
- **Entities**: Pass Types (e.g., Premium Jump Pass, VIP Jump Pass)
- **Columns**: Duration, Price, Inclusions
- **Terminology**: "Jump Pass," "All Day Pass," "Ultimate Pass"

### Relationships with Other Workbooks
The pass information is referenced in the "FAQs" and "Voicebot Scripts" workbooks to provide customers with pricing details and booking options.

## FAQs

### Purpose
The "FAQs" workbook addresses common questions from customers, offering clear answers about services, policies, and operational details. It aims to reduce customer inquiries by providing readily available information.

### Key Entities, Columns, and Terminology
- **Entities**: Questions and Answers
- **Columns**: Question, Answer
- **Terminology**: "Waiver," "Booking," "Group Discounts"

### Relationships with Other Workbooks
FAQs often reference details from the "Passes" and "Promotions" workbooks to provide comprehensive answers.

## Promotions

### Purpose
The "Promotions" workbook lists current promotional offers, including discounts on birthday parties and group bookings. It is used to inform customers about special deals and encourage bookings.

### Key Entities, Columns, and Terminology
- **Entities**: Promotion Types (e.g., Birthday Party Discounts)
- **Columns**: Description, Code, Validity Period
- **Terminology**: "Promo Code," "Discount"

### Relationships with Other Workbooks
Promotions are linked to the "Passes" and "Voicebot Scripts" workbooks for promotional pricing details.

## Voicebot Scripts

### Purpose
The "Voicebot Scripts" workbook provides pre-written responses for customer service interactions, ensuring consistency and efficiency in handling inquiries over the phone or online.

### Key Entities, Columns, and Terminology
- **Entities**: Script Types (e.g., Greeting, Closing)
- **Columns**: Scenario, Script Text, Notes
- **Terminology**: "Hold," "Transfer to Human"

### Relationships with Other Workbooks
Voicebot scripts incorporate information from all other workbooks to provide accurate responses.

## Shared Vocabulary

- **$** → "dollars"
- **min** → "minutes" or "minimum (depending on context)"
- **AM/PM** → "ante meridiem/post meridiem"
- **Aero Socks** → Required reusable socks for trampoline activities
- **VIP** → Very Important Pass

## Cross-workbook Relationships

- **Passes and Promotions**: Discounts apply to pass prices.
- **FAQs and Voicebot Scripts**: FAQs inform the content of voicebot scripts.
- **Promotions and Voicebot Scripts**: Promotion details are included in customer interactions.

## Voicebot Considerations

- **Tables**: Simplify complex tables for clarity in verbal communication.
- **Cryptic Codes**: Expand promo codes (e.g., "APRILBDAY50") to full descriptions.
- **Dense Markdown**: Break down dense information into simpler, conversational language.
- **Ambiguous Pronouns**: Avoid pronouns that could lead to confusion; use specific terms instead.

This document serves as a foundational guide for understanding the interconnected nature of AeroSports Scarborough's workbooks and supports the development of effective customer service tools.
```
